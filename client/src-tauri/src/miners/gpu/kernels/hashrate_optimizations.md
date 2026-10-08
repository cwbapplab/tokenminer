# Hashrate optimizations: what was measured, what held, what did not

Everything below was measured on this dev box, an RTX 5080 (sm_120a, 84 SMs, 64 MiB L2, CUDA 13.4),
unless a row says otherwise. Each result names the harness that produced it, so any of it can be
re-run rather than believed. Rows measured on other cards are labelled with the card.

**The one-line summary:** the search is bound by its operand path, not by bandwidth, not by
occupancy, not by registers, and not by the tensor pipe. Nothing tried so far has moved the shipping
rate. The measured ceilings and the exact list of dead ends are below, so the next attempt does not
repeat them.

---

## 1. How to measure on this box (read first — three traps, all hit for real)

1. **Spread operands, never a constant or zero fill.** This GPU compresses repeated blocks in
   memory. The same blob kernel reads **180.7 TMAC/s on a constant fill and 133.4 on a spread
   pattern** — a 40% overstatement, from data alone. Both `bench_fold.cu` and `bench_energy.cu`
   now use a full-period mix and warn on the impulse/all-ones modes.
2. **Compare within one process, never across runs.** The blob's own rate moves *a lot* between
   runs of the same binary — it read 133.8, 165.3, 176.1 and 183.2 TMAC/s across sessions, so a
   "ratio to the blob" taken from different runs can swing 0.78 to 0.93 on identical code. Only a
   ratio measured back to back in one process means anything. This is why §5e-bis quotes
   "B-pipe vs staged within one run" as the load-bearing number.
3. **A green check that cannot fail is not a check.** The fold harness copied the blob's output into
   the transcript buffer before running ours for a full day, which made "transcripts identical: yes"
   vacuously true and hid four real kernel bugs. It now poisons the buffer before *both* runs and
   requires each side to be seen writing every slot.

The rate unit is **TMAC/s**, and it equals TH/s numerically at the production geometry (one tile is
16x16x4096 = 1,048,576 MACs and the reference's bench uses `2^20/1e6`), so the two are one unit.

---

## 2. The power cap, and why energy per MAC is the metric here

The box runs the 5080 at a **252 W limit** (Current and Requested both 252 W, against a 360 W
default) — 70%, set deliberately. Under a sustained fold the card sits **at the cap**
(`SW Power Cap: Active`, max sample 252.4 W) and the clock throttles to hold it.

Consequence, and it is not a detail: at a power cap `hashrate = clock x MACs/clock`, and the clock
is whatever fits the wattage. So **hashrate is set by energy per MAC**, and a change that does the
same work for fewer joules buys back clock 1:1. Every measurement below is more useful read as
TMAC/s **per watt** than as raw TMAC/s.

Do not trust a stale reading here: NVML's `power.limit` can lag a query. `nvidia-smi -q -d POWER`'s
Current/Requested pair is authoritative.

---

## 3. The ceilings

`bench_peak.cu` — a kernel that does nothing but `mma.sync.aligned.m16n8k32` with both operands held
in registers, so no shared memory and no staging are in the loop.

| accumulators/thread | TMAC/s | notes |
|---|---|---|
| 4 | 212.4 | |
| 8 | 218.0 | |
| 12 | 219.7 | |
| 16 | 219.9 | |

**Flat from 8 up: the int8 tensor pipe on this chip issues at ~220 TMAC/s (~440 TOPS), and it is the
pipe, not instruction-level parallelism, that is the limit.** For scale, the 5080's dense fp16
tensor peak is ~450 TOPS, so int8 is *not* running at fp16's rate on sm_120 — the `m16n8k32.s8` path
is half-rate here.

Where the kernels sit against it:

| kernel | TMAC/s | % of 220 |
|---|---|---|
| Triton search blob | 133.4 | 61% |
| our hand-written fold (m=2) | 95.4 | 43% |
| peak (register-resident) | 211.7 | 96% |

---

## 4. The wall: the operand path, priced in joules

`bench_energy.cu` — same geometry as `bench_fold.cu` (131072 x 131072 x 2048), looped to steady
state, power sampled by `nvidia-smi` from outside the process.

| mode | TMAC/s | W | SM MHz | nJ/MAC |
|---|---|---|---|---|
| peak, operands in registers | 211.7 | 126 | 2596 | 0.60 |
| blob, grid **fits L2** (8K x 8K) | 156.1 | 193 | 2418 | 1.24 |
| blob, production grid (DRAM-fed) | 133.4 | 190 | 2117 | 1.43 |

Two conclusions, and they are the whole story:

- **The blob costs ~2.4x the energy per MAC of pure math** (1.43 vs 0.60 nJ/MAC), and under the cap
  it clocks 18% lower for it (2117 vs 2596 MHz). That 18% is lost hashrate.
- **The on-chip path is the larger term, not DRAM.** Shrinking the grid so only the operands change
  (everything becomes L2-resident) recovers **+26%** (133 -> 156) and 300 MHz. Cutting DRAM traffic
  instead is the smaller term (**+15%**). So the shared-memory / `ldmatrix` / barrier machinery —
  not global traffic — is most of the surplus.

The independent profile agrees: `barrier` is the top stall and `math_pipe_throttle` becomes the top
stall once the barrier count drops (`POW_BK` 32 -> 64, below).

**The model.** Per warp per 32-byte k-block, A costs `512*m` bytes of shared and B costs 8
`ldmatrix.x4` = 4096 bytes whatever the shape, because one `ldmatrix.x4` covers two eight-column
atoms. That gives **MACs per shared byte = `128m/(m+8)`**: 42.7 at m=4, 34.9 at m=3, 32.0 at m=2.
The blob sits at m=4 and our hand-written fold at m=2, which is why the blob reads B's shared bytes
4x fewer times per MAC — and why it wins on energy too.

---

## 5. Every lever tested, and its result

### 5a. Shape of the warp tile — closed

A candidate is 2 rows x 128 columns and the fold's XOR spans all 256 cells, so a warp can finish a
candidate without leaving the warp only if it owns all 128 columns. The m-extent is therefore free
and the n-extent is forced. Measured, paired against the blob in one run:

| warp tile | regs | blocks/SM | ours | blob | ratio | predicted |
|---|---|---|---|---|---|---|
| 16x128 (m=1) | 80 | 3 | 99.7 | 176.9 | — | — |
| 32x128 (m=2), B staged per warp | 153 | 3 | 125.8 | 168.7 | 0.746 | 0.80 |
| 32x128 (m=2), B spread over all 8 columns | 168 | 3 | 84.2 | 133.5 | 0.63 | 0.75 |
| 32x128 (m=2), B staged once a block | 168 | 3 | 95.4 | 133.0 | 0.72 | 0.75 |
| 48x128 (m=3), rewritten | 255 | 1 | 102.8 | 132.6 | — | **fails** |

- **m=4 is unreachable:** 256 accumulator registers against a 255 ceiling.
- **m=3 is not expressible:** a 48-row warp does not divide the 128-row tile, so warps 2 and 5
  straddle a tile boundary. It fails deterministically at candidate 64. 384 rows only moves the
  boundary. This is structural, not a bug to fix.
- **m=2 is the only shape that tiles a 128-row block** other than m=1, and it lands at 0.72 against
  the model's 0.75. The model is right; the shape is at its ceiling.

### 5b. Occupancy — dead

The blob's 211 registers cap it at 2 blocks/SM while shared memory would allow 3, so a third block
looked free. It is not.

- `CU_JIT_MAX_REGISTERS` at `cuModuleLoadDataEx` is **ignored** for this PTX, and so is
  `ptxas --maxrregcount` — both changed nothing at 168/160/128. The entry has `.reqntid 128`, an
  explicit launch bound, and that suppresses the flag. **Injecting `.maxnreg` into the PTX is the
  only knob that binds.**
- Recompiled at 168 / 160 / 128 registers (with 284 / 404 / 840 bytes of spill stores) and run:
  **77.0 / 58.5 / 29.2 TMAC/s**, against the uncapped 133.6. The third and fourth blocks lose by 1.7x
  and 4.6x.
- The control that makes this a conclusion: our hand-written m=2 fold reaches a *genuine* 3
  blocks/SM at 168 registers with **zero** spill and still only gets 95.4. The loss is the tile
  shape's operand reuse, not the spill. **Occupancy past 2 blocks cannot pay for any shape that
  does not already hold the reuse.**

### 5c. Staging-path changes — the six that are dead

Measured on the *fused* kernel at production geometry (131072², k=4096, rank=256), each
fold-checked before its timing was believed:

| lever | result |
|---|---|
| 4-lanes-per-tile BLAKE3 -> 1 | +3% |
| per-k-step `__syncthreads` removed | -3% (it helps) |
| 30 of 36 warp `LDS.32` removed (cuts 90% of shared bytes) | 0% |
| pipeline depth 2 -> 3 -> 4 | +1.6%, +0.4% |
| staging address math hoisted out of the step loop | 0% |
| `cp.async.ca` (L1) instead of `.cg` (bypass) | -4% |
| working set 67 MB -> 1074 MB staged | flat |
| `ldmatrix` (9 `LDSM`) replacing 36 `LDS.32` | +0.2% (within noise) |
| **TMA `cp.async.bulk` instead of `cp.async.cg`** | **-83%** — see below |

**TMA is not a lever on this chip — measured, do not propose it.** With identical bytes, contiguity
and in-flight depth, swapping only the engine (84 blocks x 512 threads, 12 KiB x 2048 steps,
L2-resident, 3 slices in flight):

| engine | GB/s |
|---|---|
| `cp.async.cg` (shipped) | **4636** |
| plain global -> register -> shared (control) | 3020 |
| `cp.async.bulk` (TMA) | **775** |

TMA is **5.9x slower than `cp.async`, and slower than the naive register-mediated copy.** The control
is what makes this a statement about TMA rather than about `cp.async` being fast. Consumer Blackwell
has a heavily cut-down TMA unit; datacenter Hopper/Blackwell TMA numbers do not transfer to a
GeForce part.

### 5d. The k-step width — the one staging change that landed

Removing the per-k-step `__syncthreads` outright lost 3%, but **`POW_BK` 32 -> 64 keeps the barrier
and doubles the MACs between barriers: +77%** (71.7 -> 127.1 TMAC/s, five alternating reps, ranges
71.5-73.3 against 122.4-130.5, no overlap). Shipped. The profile confirms the mechanism: barrier
stall 3.99 -> 0.92 cycles/instruction, and `math_pipe_throttle` (2.62 -> 4.12) becomes the top stall.

The barrier was never costing because it existed; it was costing because one `mma` k-block sat
between each one.

`POW_BK` is arch-conditional: 64 on the tensor path, 32 on Turing's DP4A fold (which spills 76 bytes
at 64 against 20 at 32).

### 5e. B-direct fold — the energy lever works, the port is throughput-bound

The reference's B-direct fold, built for our kernel (`FOLD_BDIRECT` in `fold_sm120.cuh`): B' is
pre-materialized in `ldmatrix` fragment order on the host, so each lane reads one `LDG.128` per
k-block straight into registers and **B never enters shared memory.** Correct — transcripts
identical, sabotage bites, shared drops 36864 -> 18432 B — and it does to the energy exactly what the
theory says:

| fold | TMAC/s | W | SM MHz |
|---|---|---|---|
| staged B (ships today) | 95.4 | 194-207 | 2165 (dips to **1815**) |
| B-direct | 56.3 | **141** | **2530** |

**-27% power, +17% clock.** But the rate fell anyway: the staged fold's `cp.async` pipeline hides
B's global latency, and reading B synchronously from global puts it back on the critical path, so
B-direct is ~2x worse *per clock*. **A naive B-direct is a loss (56 vs 95).**

#### 5e-bis. Register-pipelined B-direct (`FOLD_BPIPE`) — the win

That naive loss is *entirely* latency exposure, and register double-buffering fixes it. `FOLD_BPIPE`
keeps the next k-block's B fragments in a second register buffer: each iteration issues k-block
`kb+1`'s `LDG.128`s, then runs k-block `kb`'s `ldmatrix` + 32 mma, then rotates. The load is in
flight across a whole k-block, so the mma never waits on it. Measured in one session, all
fold-checked before their timings were believed (transcripts identical, both sides seen to write
every slot, sabotage bites):

| fold | TMAC/s | blob (co-timed) | ratio | regs | blocks/SM | W | pJ/MAC |
|---|---|---|---|---|---|---|---|
| staged B (ships) | 97.2 | 176.1 | 0.55 | 168 | 3 | 342 | 3.52 |
| B-direct, synchronous | 58.7 | 183.2 | 0.32 | 168 | 3 | 223 | 3.80 |
| **register-pipelined B-direct** | **137.0** | **176.7** | **0.78** | 214 | 2 | 345 | **2.51** |
| Triton blob | 176.1 | — | — | 211 | 2 | 339 | 1.95 |
| register-resident peak | 234.1 | — | — | — | — | 261 | 1.11 |

**Two conclusions, and both are the point of the whole exercise:**

- **+41% over the shipping hand-written fold, and −28% energy per MAC** — both axes at once, which
  is the outcome that only a real architecture change produces. Under the cap a watt saved is clock
  recovered, so at *equal* sustained power (both settle at 252 W) B-pipe does **127.3 vs 92.2
  TMAC/s, +38%** — a pure energy win, not a time-for-power trade.
- **The sync-vs-pipelined gap is 58.7 → 137.0, 2.33x.** B-direct fails only because its load sits on
  the critical path; give it a k-block of slack and the "loses 2x per clock" result (5e) inverts.
  Same instruction mix, same shared-memory footprint (18432 B), same warps.

Two implementation facts that cost real time and are worth not rediscovering:

- **`#pragma unroll 2` is load-bearing.** It is what turns the `bnext → bcur` rotate into register
  renaming; without it ptxas keeps the copy and the pipeline does not overlap.
- **`FOLD_N_ATOMS` (16) is the accumulator count, not the B-fragment count (8).** A k-block has eight
  16-column B tiles, each feeding *two* n8 atoms. Substituting `FOLD_N_ATOMS` for the B-fragment
  bound reads past the accumulator array — and it does so *plausibly*: impulse operands passed
  (every B value but k=0 is zero, so an index error is invisible), and it read **151 TMAC/s** on a
  kernel that was computing out-of-bounds garbage. Only a spread-pattern value check caught it.
  The bound is `FOLD_B_TILES`.

**Where it stands against what ships.** Production launches the **Triton blob**, not `fold_sm120`,
so the question is B-pipe vs the blob — and that ratio is **run-dependent, because the blob's own
reading is**: across sessions the blob measured 133.8, 165.3, 176.1 and 183.2 TMAC/s on the same
binary and inputs, while B-pipe stayed in 125–141. The ratio ranged **0.78 to 0.93**; in the best
run B-pipe is at *parity* with the shipping kernel. The stable, load-bearing number is
**B-pipe vs staged within one run: +35% to +41% every time.** Closing the last few percent to the
blob is the remaining work; §7 names the levers.

### 5f. Everything else — no change, or not applicable

- **Staged traffic is exhausted as a lever.** Bytes/MAC is `(T_r+T_c)/2048` for a `T_r` x `T_c` block
  of 16x16 tiles, minimised at `T_r = T_c ~ 11` -> 0.0110 against the shipped 0.0117, i.e. **6%**.
  The rectangle cannot grow further because one 16x16 tile costs 256 int32 accumulator registers and
  the whole SM register file is 65,536.
- **`cols4`** measured **64 registers, 0 spills, 2 blocks/SM, 32 warps** (up from 16) and still
  **lost 16%** to the extra traffic. Doubling occupancy does not pay for itself. **`cols2`**
  (the reference's own blocking) is **43% slower** than shipped.
- **CUDA 13 core** (sm_120 only): the same fold on the newer toolkit is **+3.2%** on all-Blackwell
  rigs — the largest single win recorded, and it is a toolchain effect, not a code change.
- **Not bandwidth-bound.** An L2-resident grid runs at the same per-SM rate as the DRAM-fed one.

### 5g. From the reference miner (super3/llmjob PR #253, v0.5.13) — what transfers

The PR is a new-card release; **it makes no change for sm_120** (its own benchmark section says so,
and its sm_120 kernels are byte-identical between the two builds it tested). Its measured wins are
all on sm_80 (A100/A30) and sm_90a (Hopper):

| change | card | measured |
|---|---|---|
| unfused fold (hash in its own kernel) | sm_80 | +2.2% to +4.8% |
| wgmma fold, 2-CTA cluster B multicast | H100 SXM / NVL | +11.1% / +3.5% |
| reduce-scatter readout | H100 | +4.46% / +4.23% |
| wgmma fold vs its own cp.async fold | H100 | 436 -> 630 TH/s |

**The one idea that transfers is the 2-CTA cluster with B multicast** — it halves B's L2 bytes, and
B's staging is exactly our wall. **And the mechanism does exist on our chip:** sm_120a supports
thread-block clusters and distributed shared memory — verified with a real 2-CTA cluster on the
5080 (`cudaDevAttrClusterLaunch = 1`, a peer CTA's shared memory read through
`cluster.map_shared_rank` returned the expected value). Two build notes for sm_120a + clusters:
use `<cooperative_groups.h>` / `cg::this_cluster()` (`__cluster_sync` is not the API), and nvcc needs
`-std=c++17 -Xcompiler /Zc:preprocessor` on Windows.

Caveat, and it is why this is a *measure first* and not a port: their gain was partly energy on
cards at their power cap (like ours), but the multicast halves **DRAM** reads while our larger term
is **on-chip** — and DSMEM reads go over the shared path at higher latency than local shared. It
may buy little here. Test it; do not port it on the strength of Hopper's number.

---

## 6. Production context

The split path's per-attempt decomposition (release build, reproduced twice), so the fold's share
of the whole is clear:

| step | time |
|---|---|
| fill + commitment + noise draw + noising (setup) | ~10.8 ms |
| transcript zero (4 GB) | 4.6 ms |
| **fold** | **214-249 ms** |
| postpass + scan | 8.4 ms |

The fold is ~90% of an attempt, so it is the only place a meaningful win can come from. The record
for the split path is **156.9 TMAC/s fold-only** (its own basis) against 133.6 end to end — the two
bases are not interchangeable.

Per share (host side, off the steady-state path): reading a 256 MiB matrix back 32 ms, the `as u8`
copy 28 ms, the Merkle tree 31 ms — and `MerkleTree::new` copies its input again, so that is two
redundant 256 MiB passes per matrix. That is per-win latency, not hashrate.

---

## 7. What is left

Ranked by expected value against the measurements above:

1. **Close B-pipe's last 0.78 → 1.0 against the blob** (5e-bis). The one axis not yet exercised is
   the warp tile: the blob uses m=4 and pays a cross-warp reduction for it (4.0 MACs per shared
   byte); our m=2 gets 3.20 (5a). At **m=2 the accumulator is 128 registers, but with B out of
   shared the register budget is different** — m=3 (192 accumulator registers) was ruled out before
   only because B's staging plus the operand working set pushed past 255. It is worth re-deriving
   that ceiling for the B-pipe shape rather than inheriting the old one. Any gain here is
   accumulator-register-bound, not shared-memory-bound.
2. **2-CTA cluster B multicast** (5g). Mechanism verified present on sm_120a; expected value lower
   than it looks because it targets DRAM, which is our smaller term. Measure before believing.
3. **Nothing in the tile shape at m≤2.** m=4 needs 256 accumulator registers, m=3 does not tile a
   128-row block. Section 5a is a proof, not a preference.

The bar for any of these is the **blob's 1.95 pJ/MAC** (this session's co-timed reading), not our
hand-written kernel's 2.51 — B-pipe has to beat the shipping kernel's energy, or it is a regression
even though it beat our own staged alternative by 28%.

**What not to try again:** occupancy/registers (5b), the six dead staging levers (5c), TMA (5c), the
tile rectangle (5f), split-k, and any bandwidth framing (5f). Each was measured; each is recorded
above with its number.

---

## 8. A note on the power reading

The box's power limit is **not fixed at 252 W**. This session `nvidia-smi -q -d POWER` reports
Current = Requested = Default = **360 W**, yet a sustained fold still settled at **252 W mean** for
both the staged fold and B-pipe, with `power.draw` excursions to 368 observed transiently. Two
things follow:

- Report energy **per MAC** (or a co-timed ratio), never "TMAC/s at the cap" against a hard-coded
  wattage — the cap is a moving target and the harness's `PEAK_WATT` default of 252 is a *label*,
  not a measurement.
- The B-pipe vs staged comparison above is drawn from two runs that both settled at 252 W, so the
  +38% is robust to whatever the limit actually is. The earlier 345 W / 342 W means were taken with
  a sampler that included cold (pre-warmup) low-power samples and are not the steady state.
