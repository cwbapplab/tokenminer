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
- **Pearl** (`miners/pearl`, ISC) — the Rust crates behind `py-pearl-mining`
  (`zk-pow` + `pearl-blake3`). The Python/pyo3 shim is dropped; its API is
  reimplemented natively in `src-tauri/src/miners/pearl.rs`. Scope note: the
  native path proves/verifies and can run a naive CPU search; production GPU
  mining lives in Pearl's Python/CUDA stack and is out of scope.

Feature flags gate each engine (both are heavy builds):

```bash
cargo check --features quantus
cargo check --features pearl
```

## Prerequisites

- Node 20+ and npm
- Rust ≥ 1.93 (see `src-tauri/rust-toolchain.toml`)
- The TokenMiner API running (default `http://localhost:5210`) — see the repo
  root `scripts/dev.ps1`

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
  src/miners/       quantus.rs, pearl.rs, manager + commands
  capabilities/     Tauri v2 permissions (http, deep-link, store, …)
```

## Notes / follow-ups

- Access/refresh tokens are currently kept in the webview's local storage. Move
  them to an OS-encrypted store (`tauri-plugin-stronghold`) before release.
- Google sign-in is not wired to a native OAuth flow yet (`POST /api/auth/google`
  accepts an ID token).
- `client/THIRD-PARTY-LICENSES.md` records the miner licenses (Apache-2.0 / ISC).
