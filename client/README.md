# TokenMiner Desktop

Tauri v2 + React desktop client for [TokenMiner](../README.md). It signs into the
TokenMiner API and runs the mining engines **natively, in-process** — no sidecar
binaries.

## What it wraps

Both miners are git submodules under `client/miners/` and are linked directly into
the Rust backend (`client/src-tauri`):

- **Quantus** (`miners/quantus-miner`, Apache-2.0) — the `miner-service` workspace
  crate plus the CPU/GPU/CUDA engines, driven in-process via
  `miner_service::run(ServiceConfig)`. Live status is read from the miner's
  Prometheus exporter.
- **Pearl** — two pieces, both in-process. The **proof layer** is the Rust crates
  behind `py-pearl-mining` (`miners/pearl`, ISC: `zk-pow` + `pearl-blake3`); the
  Python/pyo3 shim is dropped. The **search** is the CUDA core from the
  `vendor/llmjob` submodule (`earn/native/src/pearl_*.cu`), compiled from source
  by `src-tauri/build.rs` and driven over a flat C ABI (`src-tauri/native/pearl/`
  shim, `src-tauri/src/miners/pearl/` engine). The engine mines, assembles the
  `PlainProof` and re-verifies it with `zk-pow` before it goes on the wire.

  Requires the CUDA toolkit (nvcc) and MSVC at build time for the arches
  `75/80/86/89/90a/120`; without them the build warns and the client reports GPU
  unavailable at runtime rather than failing to compile.

Feature flags gate each engine (both are heavy builds):

```bash
cargo check --features quantus
cargo check --features pearl
```

## Prerequisites

- Node 20+ and npm
- Rust ≥ 1.93 (see `src-tauri/rust-toolchain.toml`)
- The TokenMiner API running (default `http://192.168.1.2:5210`) — see the repo
  root `scripts/dev.ps1`. The stack binds `0.0.0.0`, so that LAN address is what
  other machines reach it on. Override it in **Settings → API → API base URL** if
  the host's address differs.

## Setup

```powershell
# from client/
powershell ./scripts/setup.ps1   # init the miner submodules
npm install
npm run tauri dev
```

## Layout

```
src/                React app (Tailwind, no UI kit)
  lib/              api/auth/heartbeat/settings/miners/tauri adapters
  components/       Tailwind UI kit + app shell (Able Pro-style)
  pages/            Login, Dashboard, Miners, Analytics, Settings
src-tauri/          Rust backend
  src/miners/       quantus.rs, pearl/ (engine, abi, config, proof, target), manager + commands
  native/pearl/     flat C ABI shim + ABI check over the vendored llmjob CUDA core
  capabilities/     Tauri v2 permissions (http, deep-link, store, …)

vendor/llmjob/      submodule: the upstream Pearl CUDA core, built from source
```

## Notes / follow-ups

- Access/refresh tokens are currently kept in the webview's local storage. Move
  them to an OS-encrypted store (`tauri-plugin-stronghold`) before release.
- Google sign-in is not wired to a native OAuth flow yet (`POST /api/auth/google`
  accepts an ID token).
- `client/THIRD-PARTY-LICENSES.md` records the miner licenses (Apache-2.0 / ISC).
