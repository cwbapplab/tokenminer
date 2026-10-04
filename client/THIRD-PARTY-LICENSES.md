# Third-party licenses

This desktop client links two external mining projects as git submodules under
`client/miners/`. Neither is MIT — see below. Both are permissive and compatible
with redistribution; keep this file with any binary distribution.

## Quantus — `client/miners/quantus-miner`

- Project: https://github.com/Quantus-Network/quantus-miner
- Pinned: tag `v4.2.0`
- License: **Apache-2.0** (see `client/miners/quantus-miner/LICENSE`)
- The protocol crate `quantus-miner-api` (crates.io) is **MIT-0**.
- Linked crates: `miner-service`, `engine-cpu`, `engine-gpu` (wgpu), `engine-cuda`
  (cudarc), `pow-core`, `quic-transport`, `metrics`.
- Also pulls `qpow-math` from `https://github.com/Quantus-Network/chain.git`
  (tag `v0.7.1-q-day-2`) and `qp-poseidon-core` (crates.io).

## Pearl — `client/miners/pearl`

- Project: https://github.com/pearl-research-labs/pearl
- Pinned: `master` at the commit recorded by the submodule pointer.
- License: repo root is **ISC** (see `client/miners/pearl/LICENSE`). The
  `vllm-miner` package self-declares MIT; the vendored `plonky2/` tree is
  **MIT OR Apache-2.0**.
- Linked crates: `zk-pow`, `pearl-blake3`, and the vendored `plonky2`
  (`plonky2`, `plonky2_field`, `plonky2_maybe_rayon`, `starky`).
- The Python stack (`py-pearl-mining`, `miner/**`) is **not** built or shipped by
  this app; we link the Rust crates behind `py-pearl-mining` directly.

## Rust / JS dependencies

All other crates and npm packages used by the client keep their own licenses;
run `cargo license` / `npm-license-checker` in `client/src-tauri` and `client`
for a full inventory before release.
