//! Flat FFI to llmjob's Pearl CUDA core (`native/pearl/pearl_ffi.h`).
//!
//! The shim keeps the core's `std::vector`-carrying result structs on the C++
//! side and hands out borrowed slices. Every pointer in a [`PearlHitFlat`] is
//! valid only until the next call on the same handle, so [`NativeCore::collect`]
//! and [`NativeCore::next_hit`] copy everything into an owned [`Hit`] before
//! returning. Nothing here keeps a raw pointer across an FFI call.
//!
//! [`NativeCore`] is `!Send`/`!Sync` on purpose: a CUDA context is bound to the
//! thread that created it, so it must be created, driven and dropped on the one
//! miner thread (`crate::stratum::spawn_miner`).

use std::ffi::{c_char, c_int};
use std::marker::PhantomData;

use super::config::Profile;

pub const HEADER_BYTES: usize = 76;
pub const HASH_BYTES: usize = 32;

/// Mirror of `PearlProfileFlat`. Laid out field for field over the vendored
/// `PearlProfile`; `pearl_abi_check.cpp` asserts they agree at build time.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PearlProfileFlat {
    pub k: u32,
    pub rank: u16,
    pub mma_type: u16,
    pub m: u32,
    pub n: u32,
    pub seed_derivation: u32,
    pub col_batch: u32,
    pub hash_big_endian: u32,
    pub operand_fill: u32,
}

impl From<&Profile> for PearlProfileFlat {
    fn from(p: &Profile) -> Self {
        Self {
            k: p.k,
            rank: p.rank,
            mma_type: p.mma_type,
            m: p.m,
            n: p.n,
            seed_derivation: p.seed_derivation,
            col_batch: p.col_batch,
            hash_big_endian: p.hash_big_endian,
            operand_fill: p.operand_fill,
        }
    }
}

/// Mirror of `PearlSideFlat`. Pointers borrow the handle's storage.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PearlSideFlat {
    pub leaf_indices: *const u32,
    pub leaf_indices_len: u64,
    pub leaves: *const u8,
    pub leaves_len: u64,
    pub siblings: *const u8,
    pub siblings_len: u64,
    pub root: [u8; HASH_BYTES],
    pub total_leaves: u64,
}

/// Mirror of `PearlHitFlat`. Pointers borrow the handle's storage.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PearlHitFlat {
    pub jackpot_hash: [u8; HASH_BYTES],
    pub a_seed: [u8; HASH_BYTES],
    pub b_seed: [u8; HASH_BYTES],
    pub nonce: u64,
    pub salt: u64,
    pub attempts: u64,
    pub proof: *const u8,
    pub proof_len: u64,
    pub proof_a: PearlSideFlat,
    pub proof_bt: PearlSideFlat,
    pub found: c_int,
}

#[repr(C)]
pub struct PearlFfiHandle {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn pearl_ffi_select_device(
        p: *const PearlProfileFlat,
        requested: c_int,
        name: *mut c_char,
        name_len: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> c_int;

    fn pearl_ffi_create(
        p: *const PearlProfileFlat,
        device: c_int,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut PearlFfiHandle;
    fn pearl_ffi_destroy(h: *mut PearlFfiHandle);
    fn pearl_ffi_bind_thread(h: *mut PearlFfiHandle);

    fn pearl_ffi_set_job(
        h: *mut PearlFfiHandle,
        header: *const u8,
        target: *const u8,
        salt: u64,
    );
    fn pearl_ffi_reseed(h: *mut PearlFfiHandle, salt: u64);

    fn pearl_ffi_submit(
        h: *mut PearlFfiHandle,
        nonce_base: u64,
        batch: u32,
        regions_out: *mut u64,
        err: *mut c_char,
        err_len: usize,
    ) -> c_int;

    fn pearl_ffi_collect(
        h: *mut PearlFfiHandle,
        out: *mut PearlHitFlat,
        err: *mut c_char,
        err_len: usize,
    ) -> c_int;

    fn pearl_ffi_pending(h: *mut PearlFfiHandle) -> c_int;
    fn pearl_ffi_last_regions(h: *mut PearlFfiHandle) -> i64;
    fn pearl_ffi_next_hit(h: *mut PearlFfiHandle, out: *mut PearlHitFlat) -> c_int;
    fn pearl_ffi_fold_name(h: *mut PearlFfiHandle) -> *const c_char;
}

/// One side of a share proof, owned. `leaves` is `leaf_indices.len() * 1024`
/// bytes; `siblings` is a whole number of 32-byte digests.
#[derive(Debug, Clone)]
pub struct ProofSide {
    pub leaf_indices: Vec<u32>,
    pub leaves: Vec<u8>,
    pub siblings: Vec<u8>,
    pub root: [u8; HASH_BYTES],
    pub total_leaves: u64,
}

/// A hit, owned by Rust. Safe to hold across further FFI calls.
#[derive(Debug, Clone)]
pub struct Hit {
    pub jackpot_hash: [u8; HASH_BYTES],
    pub a_seed: [u8; HASH_BYTES],
    pub b_seed: [u8; HASH_BYTES],
    pub nonce: u64,
    pub salt: u64,
    pub proof_a: ProofSide,
    pub proof_bt: ProofSide,
}

const ERR_LEN: usize = 512;

fn err_buf() -> ([c_char; ERR_LEN], usize) {
    ([0 as c_char; ERR_LEN], ERR_LEN)
}

/// Read a NUL-terminated C message into a String (empty -> None).
fn read_err(buf: &[c_char]) -> Option<String> {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    if bytes.is_empty() {
        None
    } else {
        Some(String::from_utf8_lossy(&bytes).into_owned())
    }
}

unsafe fn copy_side(s: &PearlSideFlat) -> ProofSide {
    let leaves = if s.leaves.is_null() || s.leaves_len == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(s.leaves, s.leaves_len as usize).to_vec()
    };
    let leaf_indices = if s.leaf_indices.is_null() || s.leaf_indices_len == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(s.leaf_indices, s.leaf_indices_len as usize).to_vec()
    };
    let siblings = if s.siblings.is_null() || s.siblings_len == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(s.siblings, s.siblings_len as usize).to_vec()
    };
    ProofSide {
        leaf_indices,
        leaves,
        siblings,
        root: s.root,
        total_leaves: s.total_leaves,
    }
}

unsafe fn copy_hit(flat: &PearlHitFlat) -> Hit {
    Hit {
        jackpot_hash: flat.jackpot_hash,
        a_seed: flat.a_seed,
        b_seed: flat.b_seed,
        nonce: flat.nonce,
        salt: flat.salt,
        proof_a: copy_side(&flat.proof_a),
        proof_bt: copy_side(&flat.proof_bt),
    }
}

/// An owned CUDA core. Drop frees the context.
pub struct NativeCore {
    handle: *mut PearlFfiHandle,
    /// The device the context was created on, for [`NativeCore::reselect_device_for_drop`].
    device: c_int,
    // Not Send/Sync: a CUDA context is bound to its creating thread.
    _not_send: PhantomData<*mut ()>,
}

impl NativeCore {
    /// Pick a device and allocate the core for `profile`. `requested` < 0 lets
    /// the core rank the cards itself; >= 0 pins an index.
    ///
    /// Returns the chosen card's name alongside the core.
    pub fn create(profile: &Profile, requested: i32) -> Result<(Self, String), String> {
        let flat = PearlProfileFlat::from(profile);
        let mut name = [0 as c_char; ERR_LEN];
        let (mut err, err_len) = err_buf();

        let index = unsafe {
            pearl_ffi_select_device(
                &flat,
                requested,
                name.as_mut_ptr(),
                ERR_LEN,
                err.as_mut_ptr(),
                err_len,
            )
        };
        if index < 0 {
            return Err(read_err(&err).unwrap_or_else(|| "no usable CUDA device".into()));
        }

        let (mut cerr, cerr_len) = err_buf();
        let handle = unsafe { pearl_ffi_create(&flat, index, cerr.as_mut_ptr(), cerr_len) };
        if handle.is_null() {
            return Err(read_err(&cerr)
                .unwrap_or_else(|| "failed to initialise the Pearl CUDA core".into()));
        }

        let name = read_err(&name).unwrap_or_else(|| format!("GPU {index}"));

        Ok((
            Self {
                handle,
                device: index,
                _not_send: PhantomData,
            },
            name,
        ))
    }

    /// Bind the calling thread to this core's device. Call once on the miner
    /// thread before driving the handle.
    pub fn bind_thread(&self) {
        unsafe { pearl_ffi_bind_thread(self.handle) };
    }

    /// The CUDA device index the core was created on.
    pub fn device(&self) -> c_int {
        self.device
    }

    /// Which fold this card runs (for the startup log).
    pub fn fold_name(&self) -> String {
        let p = unsafe { pearl_ffi_fold_name(self.handle) };
        if p.is_null() {
            return "unresolved".into();
        }
        let bytes = unsafe { std::ffi::CStr::from_ptr(p) }.to_bytes();
        String::from_utf8_lossy(bytes).into_owned()
    }

    /// Load a job. `target` is the scaled share bound, 32 bytes big-endian.
    pub fn set_job(&mut self, header: &[u8; HEADER_BYTES], target: &[u8; HASH_BYTES], salt: u64) {
        unsafe { pearl_ffi_set_job(self.handle, header.as_ptr(), target.as_ptr(), salt) };
    }

    /// Re-draw the operands under a new salt. One salt is worth `m*n` regions.
    pub fn reseed(&mut self, salt: u64) {
        unsafe { pearl_ffi_reseed(self.handle, salt) };
    }

    /// Queue a batch without waiting. Returns the region count it covers.
    pub fn submit(&mut self, nonce_base: u64, batch: u32) -> Result<u64, String> {
        let mut regions: u64 = 0;
        let (mut err, err_len) = err_buf();
        let rc = unsafe {
            pearl_ffi_submit(
                self.handle,
                nonce_base,
                batch,
                &mut regions,
                err.as_mut_ptr(),
                err_len,
            )
        };
        match rc {
            1 => Ok(regions),
            0 => Err("submit: search pipeline full or no job".into()),
            _ => Err(read_err(&err).unwrap_or_else(|| "CUDA submit failed".into())),
        }
    }

    /// Wait for the oldest queued batch. `Some(hit)` when it found a share. The
    /// batch's region count is read separately via [`NativeCore::last_regions`],
    /// because it is meaningful on a miss too.
    pub fn collect(&mut self) -> Result<Option<Hit>, String> {
        let mut out: PearlHitFlat = unsafe { std::mem::zeroed() };
        let (mut err, err_len) = err_buf();
        let rc =
            unsafe { pearl_ffi_collect(self.handle, &mut out, err.as_mut_ptr(), err_len) };
        match rc {
            1 => Ok(Some(unsafe { copy_hit(&out) })),
            0 => Ok(None),
            _ => Err(read_err(&err).unwrap_or_else(|| "CUDA collect failed".into())),
        }
    }

    /// The collected batch's further hits, after the one `collect` returned.
    pub fn next_hit(&mut self) -> Option<Hit> {
        let mut out: PearlHitFlat = unsafe { std::mem::zeroed() };
        let rc = unsafe { pearl_ffi_next_hit(self.handle, &mut out) };
        if rc == 1 {
            Some(unsafe { copy_hit(&out) })
        } else {
            None
        }
    }

    /// Region count of the most recent [`NativeCore::collect`], hit or miss.
    ///
    /// The core hands this back only through its hit result, so a caller reading
    /// it from `collect`'s return alone counts just the batches that found a
    /// share — which at a pool's difficulty is close to none of them, and makes a
    /// busy card look idle. `None` before the first collect.
    pub fn last_regions(&self) -> Option<u64> {
        let n = unsafe { pearl_ffi_last_regions(self.handle) };
        if n < 0 {
            None
        } else {
            Some(n as u64)
        }
    }

    pub fn pending(&self) -> i32 {
        unsafe { pearl_ffi_pending(self.handle) }
    }
}

impl Drop for NativeCore {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { pearl_ffi_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flat profile must be exactly nine packed scalars with no padding, or
    /// the reinterpret_cast to the core's PearlProfile reads the wrong bytes.
    ///
    /// 4 + 2 + 2 + 4*6 = 32, and 32 is what the vendored `PearlProfile` measures
    /// (`pearl_abi_check.cpp` prints it). An earlier value of 36 here was simply
    /// a mis-add — every field offset below was right, and the total disagreed
    /// with the struct the assert exists to mirror.
    #[test]
    fn the_flat_profile_is_packed_and_ordered() {
        assert_eq!(std::mem::size_of::<PearlProfileFlat>(), 32);
        assert_eq!(std::mem::align_of::<PearlProfileFlat>(), 4);
        assert_eq!(std::mem::offset_of!(PearlProfileFlat, k), 0);
        assert_eq!(std::mem::offset_of!(PearlProfileFlat, rank), 4);
        assert_eq!(std::mem::offset_of!(PearlProfileFlat, mma_type), 6);
        assert_eq!(std::mem::offset_of!(PearlProfileFlat, m), 8);
        assert_eq!(std::mem::offset_of!(PearlProfileFlat, n), 12);
        assert_eq!(std::mem::offset_of!(PearlProfileFlat, seed_derivation), 16);
        assert_eq!(std::mem::offset_of!(PearlProfileFlat, col_batch), 20);
        assert_eq!(std::mem::offset_of!(PearlProfileFlat, hash_big_endian), 24);
        assert_eq!(std::mem::offset_of!(PearlProfileFlat, operand_fill), 28);
    }
}
