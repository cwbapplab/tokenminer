//! The GPU engine: llmjob's CUDA core, driven from here.
//!
//! This is the replacement for the hand-rolled cudarc/cubin/Triton stack. The
//! search itself is the vendored core; what lives here is the host loop that
//! drives it — the salt walk, the job switch, the two-deep submit/collect
//! pipeline, the share-proof assembly and the hashrate accounting — a Rust
//! transcription of `vendor/llmjob/earn/native/src/pearl_core.cc::SearchLoop`
//! and its JS host.
//!
//! The shape the rest of the client sees (`open` / `search` / `tiles` /
//! `mining`) is unchanged from the engine it replaced, so `stratum` sees
//! different types, not a different contract.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use primitive_types::U256;
use zk_pow::api::proof::IncompleteBlockHeader;
use zk_pow::ffi::plain_proof::PlainProof;

use super::abi::{Hit, NativeCore};
use super::config::{
    build_config52, job_key, seed_derivation_for, Profile, COLS_PATTERN, ROWS_PATTERN,
    SEED_SALTED,
};
use super::oracle::{self, TINY_VECTOR};
use super::proof;
use super::target;
use crate::miners::pearl_mining::PearlMining;

/// Batches queued on the device at once. Two is enough: one runs while the next
/// waits behind it. The core itself holds exactly two slots (`Ctx::kSlots`), so a
/// third submit is refused.
const DEPTH: usize = 2;

/// How long one `search` may run before returning to the driver.
///
/// llmjob's own loop runs until the job changes; the driver here also uses this
/// as the cadence of its hashrate window, and it bounds how long a stale search
/// can keep the loop before the reader notices a superseded job.
const MAX_SECONDS_PER_SEARCH: u64 = 5;

/// Owns the device-side state for one mining configuration and the search loop
/// that drives it.
pub struct GpuMiner {
    /// The CUDA core. Not `Send`; created and dropped on the miner thread.
    core: NativeCore,
    device: String,
    fold: String,
    profile: Profile,
    /// Region offsets searched since the engine started, for the hashrate.
    tiles: u64,
    /// This card's slice of the search space. One card per core, so a single
    /// card is `(0, 1)` and walks salts 0, 1, 2, ….
    salt: u64,
    salt_stride: u64,
    /// The region offset the next queued batch starts at.
    ///
    /// It **persists across `search` calls** and resets only on a job change or a
    /// salt wrap, mirroring llmjob's `SearchLoop` (`pearl_core.cc:391,483,504`).
    /// A `search` that ends with a hit returns early — before the slice deadline —
    /// so a cursor local to `search` would start the next call at region 0 under
    /// the *same* operands, deterministically re-find the share it just sent, and
    /// submit it again. That is not a rate, it is a loop of duplicates, and the
    /// pool rejects every repeat after the first.
    nonce: u64,
    /// The job currently loaded into the core, if any. A change here — and only
    /// a change — triggers a fresh `set_job`, which re-draws the operands.
    loaded: Option<[u8; 32]>,
    /// `blake3(header ‖ config52)` for the loaded job: the key both Merkle
    /// proofs are built against.
    job_key: [u8; 32],
    /// Shares the core already found in a collected batch, after the first. Held
    /// so a batch that held two shares does not lose one.
    pending_shares: VecDeque<PlainProof>,
}

impl GpuMiner {
    /// Opens a device and prepares it to mine at this configuration.
    ///
    /// Runs the startup parity gate first: a device that cannot reproduce the
    /// vendored stack's own frozen vectors must not be allowed to produce
    /// shares. Failure returns `Err`, which the driver reports as "GPU
    /// unavailable" exactly as the previous engine did.
    pub fn open(mining: PearlMining) -> Result<Self, String> {
        let profile = profile_from(&mining, SEED_SALTED);
        check_profile(&profile)?;

        // The gate runs against a small, self-contained profile on a core of its
        // own, so a failure costs a few megabytes rather than the gigabytes the
        // mining profile's context holds. The gate profile's m/n do not enter
        // job_key, and the gate is a self-consistency check, not a comparison
        // against a captured vector, so it is valid at any runnable geometry.
        {
            let gate_profile = gate_profile();
            let (mut gate, _) = NativeCore::create(&gate_profile, -1)
                .map_err(|e| format!("could not open a CUDA device for the startup gate: {e}"))?;
            run_startup_gate(&mut gate, &gate_profile)?;
        }

        let (core, device) = NativeCore::create(&profile, -1)?;
        core.bind_thread();
        let fold = core.fold_name();

        let miner = Self {
            core,
            device,
            fold,
            profile,
            tiles: 0,
            salt: 0,
            salt_stride: 1,
            nonce: 0,
            loaded: None,
            job_key: [0; 32],
            pending_shares: VecDeque::new(),
        };

        log::info!(
            "pearl gpu: mining on {} ({} fold, m={} n={} k={} rank={}, {:.1} GiB per operand)",
            miner.device,
            miner.fold,
            miner.profile.m,
            miner.profile.n,
            miner.profile.k,
            miner.profile.rank,
            ((miner.profile.m as u64) * (miner.profile.k as u64)) as f64 / (1u64 << 30) as f64
        );

        Ok(miner)
    }

    /// The configuration the proof declares, which is the profile the core
    /// actually mines with. `stratum` reads [`PearlMining`] from its own config
    /// and only needs the resolved protocol profile from here, for the share bound.
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    /// Region offsets searched so far, used to report hashrate.
    pub fn tiles(&self) -> u64 {
        self.tiles
    }

    /// Searches `header` for a proof whose jackpot is at or below `bound`.
    ///
    /// Returns the proof for the first candidate that clears the bound, and
    /// `None` when `stop` says the job has been superseded or the time slice
    /// runs out — the normal way this ends, since a new `mining.notify` arrives
    /// far more often than a share does.
    pub fn search(
        &mut self,
        header: &IncompleteBlockHeader,
        cert_version: u32,
        bound: U256,
        stop: &dyn Fn() -> bool,
    ) -> Result<Option<PlainProof>, String> {
        // The certificate version selects the seed derivation, and a proof built
        // under the wrong one will not verify. The core takes this at
        // construction, so the context is rebuilt in place — cheap next to the
        // parity gate, and it happens at most once per session.
        let derivation = seed_derivation_for(cert_version);
        if self.profile.seed_derivation != derivation {
            self.rebuild(derivation)?;
        }

        let header_bytes = header.to_bytes();
        let key = job_key(&header_bytes, &build_config52(&self.profile));

        // A share the previous batch found and did not report yet is still valid
        // as long as the job has not moved.
        if self.loaded == Some(key) {
            if let Some(share) = self.pending_shares.pop_front() {
                return Ok(Some(share));
            }
        } else {
            self.pending_shares.clear();
            self.drain();
            let mut target_bytes = [0u8; 32];
            bound.to_big_endian(&mut target_bytes);
            self.core.set_job(&header_bytes, &target_bytes, self.salt);
            self.loaded = Some(key);
            self.job_key = key;
            // A new job draws fresh operands, so the region cursor restarts.
            self.nonce = 0;
        }

        let span = target::regions_per_salt(&self.profile);
        let deadline = Instant::now() + Duration::from_secs(MAX_SECONDS_PER_SEARCH);
        let mut queued = 0usize;

        loop {
            while queued < DEPTH {
                match self.core.submit(self.nonce, 0) {
                    Ok(regions) => {
                        queued += 1;
                        self.nonce += regions;
                        if self.nonce >= span {
                            // One salt offers m*n regions and no more: past that
                            // the search repeats itself, so the operands are
                            // re-drawn and the region index starts over.
                            self.salt += self.salt_stride;
                            self.core.reseed(self.salt);
                            self.nonce = 0;
                        }
                    }
                    // The pipeline is full despite the depth guard, or the core
                    // refused the batch. Stop filling and drain what is queued.
                    Err(_) => break,
                }
            }

            if queued == 0 {
                return Ok(None);
            }

            queued -= 1;
            match self.core.collect() {
                Ok(Some((hit, attempts))) => {
                    self.tiles += attempts;
                    // A batch can hold more than one hit; each is its own share.
                    let mut hits = vec![hit];
                    while let Some(more) = self.core.next_hit() {
                        hits.push(more);
                    }
                    for hit in hits {
                        if let Some(proof) = self.proof_for(&hit) {
                            self.pending_shares.push_back(proof);
                        }
                    }
                    if let Some(share) = self.pending_shares.pop_front() {
                        return Ok(Some(share));
                    }
                }
                Ok(None) => {}
                Err(error) => return Err(error),
            }

            if stop() || Instant::now() >= deadline {
                return Ok(None);
            }
        }
    }

    /// Build a wire proof for a hit, or `None` when it cannot be certified.
    ///
    /// A hit the core reports but this refuses is not an error: a proof that
    /// will not verify at the pool is worth strictly less than no share at all,
    /// so the caller simply hands the next one up. `build_share_proof` does the
    /// Merkle re-check and the leaf-index check; the driver runs the full
    /// verifier on whatever this returns before it goes on the wire.
    fn proof_for(&self, hit: &Hit) -> Option<PlainProof> {
        proof::build_share_proof(
            &self.job_key,
            &self.profile,
            hit.nonce,
            &hit.proof_a,
            &hit.proof_bt,
        )
    }

    /// Rebuild the context on the same device under a new seed derivation.
    fn rebuild(&mut self, derivation: u32) -> Result<(), String> {
        self.profile = self.profile.with_seed_derivation(derivation);
        let (core, _) = NativeCore::create(&self.profile, self.core.device())?;
        core.bind_thread();
        self.core = core;
        self.loaded = None;
        self.pending_shares.clear();
        Ok(())
    }

    /// Collect everything still queued, unreported, before a job switch.
    fn drain(&mut self) {
        while self.core.pending() > 0 {
            if self.core.collect().is_err() {
                break;
            }
        }
    }
}

impl Drop for GpuMiner {
    fn drop(&mut self) {
        // Leave the device idle before the context goes, so a queued batch is
        // not abandoned half-run. `pearl_ffi_destroy` frees the device memory next.
        self.drain();
    }
}

/// The profile the startup gate mines with: `k`/`rank` are the mandated pair
/// (the fold is compiled for nothing else), and `m`/`n` are the smallest that
/// keep both chunk counts powers of two and clear the fold's `n >= 8k` and
/// whole-column-span floors. Small on purpose — the gate runs at every startup
/// and its context should cost megabytes, not the mining profile's gigabytes.
fn gate_profile() -> Profile {
    Profile {
        k: 2048,
        rank: 128,
        mma_type: 0,
        m: 1024,
        n: 16_384,
        seed_derivation: SEED_SALTED,
        col_batch: 16,
        hash_big_endian: 0,
        operand_fill: super::config::OPERAND_CONST,
        rows: ROWS_PATTERN,
        cols: COLS_PATTERN,
    }
}

/// Build the mining profile from the client's configuration.
///
/// The tile patterns are protocol constants and come from `config`, not from the
/// client's JSON: the core hardcodes them, the verifier rebuilds them from a
/// proof's row indices, and a JSON value that disagreed would mine a tile no
/// pool would accept.
fn profile_from(mining: &PearlMining, derivation: u32) -> Profile {
    Profile {
        k: mining.k as u32,
        rank: mining.rank,
        mma_type: 0,
        m: mining.m as u32,
        n: mining.n as u32,
        seed_derivation: derivation,
        col_batch: super::config::PROFILE.col_batch,
        hash_big_endian: super::config::PROFILE.hash_big_endian,
        operand_fill: super::config::PROFILE.operand_fill,
        rows: ROWS_PATTERN,
        cols: COLS_PATTERN,
    }
}

/// Refuse a profile the core cannot run, with a sentence about why.
///
/// Every clause here is a way the core would otherwise fail deep inside a kernel
/// launch — or, worse, compute a wrong root in silence. See `pearl_host_create`'s
/// own checks, which these mirror.
fn check_profile(profile: &Profile) -> Result<(), String> {
    let folded = &super::config::PROFILE;
    if profile.k != folded.k || profile.rank != folded.rank {
        return Err(format!(
            "the fold kernel is built for k={} rank={}; this configuration is k={} rank={}",
            folded.k, folded.rank, profile.k, profile.rank
        ));
    }

    // The commitment folds the chunk tree as a balanced pairwise reduction,
    // which is only BLAKE3's tree when the chunk count is a power of two.
    for (name, dim) in [("m", profile.m), ("n", profile.n)] {
        let chunks = (dim as u64) * (profile.k as u64) / 1024;
        if chunks == 0 || !chunks.is_power_of_two() {
            return Err(format!(
                "{name}*k/1024 must be a power of two (got {chunks}): the commitment tree fold \
                 assumes it"
            ));
        }
    }

    // The verifier's public-params check needs k >= 16*rank, and the fold's
    // noise layout needs n large enough to hold a rank-wide strip.
    let min_n = 8 * profile.k;
    if profile.n < min_n {
        return Err(format!(
            "n must be at least 8*k = {min_n} for the fold's noise layout (got {})",
            profile.n
        ));
    }

    Ok(())
}

/// The startup gate: refuse to mine unless the stack agrees with itself.
///
/// Every failure this miner can have is silent — a wrong `config52` byte, a keyed
/// where an unkeyed hash belonged, a swapped seed, a fold that resets its
/// accumulator per chunk, a jackpot compared the wrong way round — each runs at
/// full speed, reports a healthy hashrate and simply never earns. This turns
/// "wrong" into "wrong here".
///
/// It is deliberately **not** a replay of llmjob's captured device vectors.
/// Those were captured at `k = 512, rank = 32`, and `pearl_host_submit` refuses
/// any k/rank but the mandated pair, so the vectors cannot reach the real core
/// at all — they pin the JS reference and nothing else. What is left is a live
/// self-test: mine a job whose bound admits everything, and check the core's own
/// answers against the host-side arithmetic the miner will use on every hit.
///
/// The design is the point. Inverting the acceptance test (all-ones secret
/// key, all-ones bound) makes the *first* batch a hit under any implementation,
/// correct or broken, so this stays cheap and still exercises the whole path:
/// submit, collect, the hit's Merkle proofs against `job_key`, and the salt walk.
fn run_startup_gate(core: &mut NativeCore, profile: &Profile) -> Result<(), String> {
    // The CPU half first: it costs nothing and pins `config52` / the unkeyed
    // job_key, which every downstream value depends on.
    if oracle::tiny_job_key() != TINY_VECTOR.job_key {
        return Err(
            "the ported config52/job_key does not match the reference vectors — refusing to mine"
                .into(),
        );
    }

    let header = oracle::header_all_ones();
    let key = job_key(&header, &build_config52(profile));

    // A bound of all ones admits every candidate, so the core reports a hit on
    // the first batch whatever it computes. The check is then not "did it find
    // something" — that is guaranteed — but "do its two answers agree with each
    // other": the hash the core read, and the proofs it captured for it.
    let bound = U256::max_value();
    let mut target = [0u8; 32];
    bound.to_big_endian(&mut target);
    core.set_job(&header, &target, 0);

    let span = target::regions_per_salt(profile);
    let mut draws = Vec::new();

    // Two salts is enough to tell a varying walk from a stuck one, and the
    // second is a restamp rather than a fresh draw, so it also exercises the
    // reseed path the miner relies on when a salt's regions run out.
    for salt in 0..2u64 {
        if salt > 0 {
            core.reseed(salt);
        }

        let mut nonce = 0u64;
        let mut hit = None;
        while nonce < span {
            // `submit` reports an error rather than a zero-width batch, so a
            // successful call always advances by at least one region.
            let batch = core.submit(nonce, 0)?;
            if let Some((found, _)) = core.collect()? {
                hit = Some(found);
                break;
            }
            nonce += batch;
        }

        let Some(hit) = hit else {
            return Err(format!(
                "the core returned no hit at salt {salt} against an all-ones bound — it is not \
                 searching, and it would have mined nothing while reporting a hashrate"
            ));
        };

        // The core's own comparison, re-run on the host: the jackpot is read
        // little-endian and compared to the bound. A disagreement here is one
        // side reading the hash backwards, which costs every share and raises no
        // error anywhere.
        if !target::meets_target(&hit.jackpot_hash, bound) {
            return Err(
                "the core reported a hit whose jackpot the host-side comparison rejects: the two \
                 read the hash the wrong way round"
                    .into(),
            );
        }
        if hit.nonce >= span {
            return Err(format!(
                "the core reported a hit at region {}, outside the salt's {span} regions",
                hit.nonce
            ));
        }
        // The hit must name the salt it was found under. A hit stamped with a
        // different salt would build a proof from the wrong operand draw, so its
        // Merkle roots would not match the region it declares.
        if hit.salt != salt {
            return Err(format!(
                "the core stamped the hit with salt {}, not {salt} — the salt walk is wrong",
                hit.salt
            ));
        }

        // The share the miner would actually send. `build_share_proof` re-derives
        // both Merkle roots against `job_key` and checks the leaf indices cover
        // the tile the region names, so a core whose proofs do not describe the
        // hash it reported fails here rather than at the pool.
        if proof::build_share_proof(&key, profile, hit.nonce, &hit.proof_a, &hit.proof_bt).is_none()
        {
            return Err(
                "the core reported a hit whose Merkle proofs do not verify against job_key — the \
                 share it produced could never be accepted"
                    .into(),
            );
        }

        draws.push((hit.a_seed, hit.b_seed));
    }

    // A redraw has to move the operands. Identical seeds across salts is the
    // collapsed-search failure: it mines the same tile for ever, at full speed,
    // and reports a healthy hashrate while finding nothing new.
    if draws.first() == draws.last() {
        return Err(format!(
            "the salt walk drew the same operands twice (a_seed {}) — the operands are not being \
             re-drawn",
            hex(&draws[0].0)
        ));
    }

    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The on-device half of the gate: the real core, at a real profile, driven
    /// through the same FFI the miner uses.
    ///
    /// Ignored by default because it opens the GPU and allocates a context, which
    /// a plain `cargo test` on a build host without CUDA must not do. On the dev
    /// box:
    ///
    ///     cargo test --features pearl -- --ignored the_gpu
    ///
    /// What it can and cannot prove is worth stating, because the obvious version
    /// of this test is the one that proves nothing. llmjob's captured
    /// RTX-4090 vectors (the k=512/rank=32 profile in `minerDeviceParity.test.js`)
    /// **cannot be replayed through this core**: `pearl_host_submit` refuses any
    /// k/rank but the mandated pair, so those vectors only ever exercise the JS
    /// reference now. What is left is the property the vectors were there to
    /// protect — that the device computes what we believe it computes — and the
    /// two halves of it that are checkable from outside the kernel:
    ///
    ///   * `meets_target` agrees with the core's own comparison. This is the
    ///     endianness check: the core reports a hit only when the jackpot read
    ///     little-endian clears the bound it was handed, and if our host-side
    ///     re-check disagrees with that on any hit, one of the two is wrong.
    ///   * the salt walk varies. Different salts must draw different operands
    ///     and report different seeds; a search that stopped varying is silent
    ///     (it reports a healthy hashrate and finds nothing).
    ///
    /// The Merkle/transcript agreement is covered by
    /// `proof::build_share_proof`, which refuses a hit whose proofs do not verify
    /// against `job_key` — this test asserts the hit it finds actually produces a
    /// share.
    #[test]
    #[ignore = "opens the GPU and the vendored core; run on the dev box"]
    fn the_gpu_finds_a_verifiable_share_and_the_salt_walk_varies() {
        let profile = gate_profile();
        check_profile(&profile).expect("the gate profile must be runnable");

        let (mut core, device) = NativeCore::create(&profile, -1)
            .expect("a CUDA device must be available on the box this test is run on");
        core.bind_thread();

        // The same live self-test the miner runs at startup, driven through the
        // same FFI — a green run here is a green startup gate.
        run_startup_gate(&mut core, &profile).expect("the startup gate must pass on a working card");

        eprintln!(
            "on-device gate passed on {device} ({} fold)",
            core.fold_name()
        );
    }

    /// The whole driver path, end to end: open the shipped engine, mine a share,
    /// and hand it to the same `zk-pow` verifier the pool runs.
    ///
    /// This is the test the startup gate cannot be. The gate proves the *core*
    /// is self-consistent; this proves the *driver* — the job switch, the salt
    /// walk, the tile the proof declares and the Merkle assembly — produces
    /// something the verifier accepts at the geometry we actually ship, which is
    /// the only thing standing between a working card and an accepted share.
    ///
    /// The share target is `U256::MAX`: the verifier saturates its bound to every
    /// digest, so the first region the search covers is reported. What is on trial
    /// is the assembly, not the difficulty.
    ///
    ///     cargo test --features pearl -- --ignored the_driver
    #[test]
    #[ignore = "opens the GPU, allocates a gigabyte context, and searches; run on the dev box"]
    fn the_driver_mines_a_share_the_verifier_accepts() {
        let mut miner = GpuMiner::open(PearlMining::default())
            .expect("the shipped profile must open on the dev box's card");

        let header = IncompleteBlockHeader {
            version: 0x2000_0000,
            prev_block: [0x11; 32],
            merkle_root: [0x22; 32],
            timestamp: 1_700_000_000,
            nbits: 0,
        };

        let proof = miner
            .search(&header, 3, U256::max_value(), &|| false)
            .expect("the search must not fault")
            .expect("an all-ones bound admits every region, so the first batch is a share");

        proof::verify_share_locally(&header, 3, &proof, U256::max_value()).expect(
            "the verifier the pool runs must accept the share the engine built — if this fails the \
             engine would mine silently and never earn",
        );

        // And the wire encoding round-trips through the reference decoder, so the
        // share the pool receives is the share the verifier just checked.
        let encoded = proof::encode_plain_proof(&proof, false).expect("the proof must encode");
        let decoded = {
            use base64::Engine as _;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(&encoded)
                .expect("the encoding must be valid base64");
            PlainProof::deserialize_compat(&bytes).expect("the pool's decoder must accept it")
        };
        assert_eq!(decoded.k, proof.k);
        assert_eq!(decoded.m, proof.m);
        assert_eq!(decoded.n, proof.n);

        // The same again under a different certificate version. A legacy
        // certificate makes `search` rebuild the context in place (the seed
        // derivation is fixed at construction), so this is the only place that
        // path is exercised — and a rebuild that left the old derivation behind
        // would mint shares the legacy verifier rejects, silently.
        let legacy = miner
            .search(&header, 2, U256::max_value(), &|| false)
            .expect("the rebuild must not fault")
            .expect("an all-ones bound admits every region");
        proof::verify_share_locally(&header, 2, &legacy, U256::max_value())
            .expect("a share mined under a legacy certificate must verify under one");

        eprintln!(
            "driver e2e: {} fold mined a share the verifier accepts on {} ({} bytes on the wire)",
            miner.fold,
            miner.device,
            encoded.len()
        );
    }

    /// Two searches on an unchanged job must not report the *same* region.
    ///
    /// This is the duplicate-share loop: a hit makes `search` return early, before
    /// its slice deadline, so if the region cursor were local to `search` the next
    /// call would start at region 0 under unchanged operands, recompute the same
    /// jackpot and resend the share it just sent. On a live job that produced a
    /// "solution found" every ~0.3 s — a steady stream of identical submissions.
    ///
    /// The bound is `U256::MAX`, so *every* region is a hit and each call returns
    /// its first covered region; the cursor therefore has to be visible as a
    /// different proof, not merely a different count. The encoded proofs are
    /// compared rather than the structs because the region reaches the wire as the
    /// leaf indices it selects, which is exactly what the pool rejects as a
    /// duplicate.
    ///
    ///     cargo test --features pearl -- --ignored advance
    #[test]
    #[ignore = "opens the GPU, allocates a gigabyte context, and searches; run on the dev box"]
    fn consecutive_searches_on_one_job_advance_past_the_region_they_reported() {
        let mut miner = GpuMiner::open(PearlMining::default())
            .expect("the shipped profile must open on the dev box's card");

        let header = IncompleteBlockHeader {
            version: 0x2000_0000,
            prev_block: [0x33; 32],
            merkle_root: [0x44; 32],
            timestamp: 1_700_000_001,
            nbits: 0,
        };

        let first = miner
            .search(&header, 3, U256::max_value(), &|| false)
            .expect("the search must not fault")
            .expect("an all-ones bound admits every region");
        let second = miner
            .search(&header, 3, U256::max_value(), &|| false)
            .expect("the search must not fault")
            .expect("an all-ones bound admits every region");

        let encoded = |proof: &PlainProof| {
            proof::encode_plain_proof(proof, false).expect("the proof must encode")
        };
        assert_ne!(
            encoded(&first),
            encoded(&second),
            "the second search re-reported the region the first already sent — the region cursor \
             is not advancing, so the pool is being fed the same share over and over"
        );
    }

    /// The shipped default must be a profile the core can actually run. Each
    /// clause is checked by breaking it, because a `check_profile` that accepted
    /// everything would pass this test vacuously.
    #[test]
    fn the_shipped_profile_is_accepted_and_broken_ones_are_refused() {
        let good = profile_from(&PearlMining::default(), SEED_SALTED);
        check_profile(&good).expect("the shipped default must be runnable");

        // A k the fold is not built for.
        assert!(check_profile(&Profile { k: 4096, ..good }).is_err());
        // A rank the fold is not built for.
        assert!(check_profile(&Profile { rank: 256, ..good }).is_err());
        // m that is not a power-of-two chunk count.
        assert!(check_profile(&Profile { m: good.m + 1024, ..good }).is_err());
        // n below the fold's k >= 16*rank floor.
        assert!(check_profile(&Profile { n: 4096, ..good }).is_err());
    }

    /// `profile_from` must take the geometry from the mining config but the
    /// protocol constants from `config`: a client JSON that carried a different
    /// tile would otherwise mine a tile no pool accepts.
    #[test]
    fn the_profile_takes_its_patterns_from_the_protocol_constants() {
        let mining = PearlMining {
            rows_pattern: vec![0, 1],
            cols_pattern: (0..128).collect(),
            ..PearlMining::default()
        };
        let profile = profile_from(&mining, SEED_SALTED);
        assert_eq!(profile.rows, ROWS_PATTERN);
        assert_eq!(profile.cols, COLS_PATTERN);
        assert_eq!(profile.tile_size(), 256);
    }

    /// The engine's region walk and the proof's tile must agree, or a share is
    /// built for a different tile than the hash certified.
    #[test]
    fn a_region_names_the_tile_the_proof_declares() {
        let profile = profile_from(&PearlMining::default(), SEED_SALTED);
        let tile = target::region_to_tile(0, &profile);
        assert_eq!(tile.rows, ROWS_PATTERN.to_vec());
        assert_eq!(tile.cols, COLS_PATTERN.to_vec());
        assert_eq!(target::regions_per_salt(&profile), 8192 * 16384);
    }

    /// The hashrate unit is unchanged by the port: one region is one
    /// `rows x cols x k` tile, and the penalized adjustment factor is the same
    /// 524288 the driver has always scaled a share target by.
    #[test]
    fn the_share_factor_is_unchanged_by_the_port() {
        let profile = profile_from(&PearlMining::default(), SEED_SALTED);
        assert_eq!(
            crate::miners::pearl::target::penalized_adjustment_factor_u64(&profile),
            524_288
        );
    }
}
