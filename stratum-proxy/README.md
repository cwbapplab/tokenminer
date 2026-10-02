# Stratum proxy

A transparent Stratum relay that sits between miners and their pool, and reports the shares the
pool **accepts** back to the TokenMiner API.

```
miner ──▶ nginx (TLS) ──▶ stratum proxy ──▶ pool
                                │
                                └── accepted share ──▶ POST /internal/mining/shares
```

## What it does

1. Accepts a connection and waits for `mining.authorize`.
2. Reads the worker identity from the username (`wallet.worker`) and resolves it against the API.
   A worker with no active session is refused, so the proxy never relays for an unknown device.
3. Opens a connection to that worker's pool and relays every message **verbatim**, in both
   directions. The miner still receives exactly what the pool sent.
4. Watches `mining.submit` requests and their replies. When the pool accepts a submission it
   reports the share to the API with a signed request. Rejected shares are relayed and dropped —
   nothing is reported.

The `.NET` API is never in the Stratum path: the proxy talks to it only to resolve a worker and
to report accepted shares, both off the critical relay path.

## Layout

| Crate | Responsibility |
|---|---|
| `protocol` | Stratum JSON-RPC line framing, message parsing, `mining.submit` arguments |
| `api-client` | Signed HTTP client for the internal API (config, worker resolution, shares) |
| `proxy` | Listener, worker identification, upstream connection, bidirectional relay |

## Configuration

All configuration comes from the environment.

| Variable | Default | Meaning |
|---|---|---|
| `TOKENMINER_API_BASE_URL` | *required* | API base URL, e.g. `http://localhost:5210` |
| `TOKENMINER_SERVICE_SECRET` | *required* | Must equal the API's `Mining:ServiceSharedSecret` |
| `TOKENMINER_SERVICE_ID` | `stratum-proxy` | Service identity used in replay protection |
| `STRATUM_LISTEN_ADDR` | `0.0.0.0:3333` | Address miners connect to |
| `CONFIG_CACHE_TTL_SECONDS` | `60` | How long the pool routing table is cached |
| `WORKER_CACHE_TTL_SECONDS` | `30` | How long a worker resolution is cached (including misses) |
| `AUTHORIZE_TIMEOUT_SECONDS` | `30` | How long a client may stay connected without authorizing |
| `RUST_LOG` | `info` | Tracing filter |

## Running

```bash
export TOKENMINER_API_BASE_URL=http://localhost:5210
export TOKENMINER_SERVICE_SECRET=<same value as Mining:ServiceSharedSecret in the API>
cargo run --release -p sp-proxy
```

Then point a miner at it. The API hands out a working command from `POST /api/mining/start`; set
`Mining:PublicStratumEndpoint` (e.g. `your-host:3333`) so generated commands target the proxy
rather than the pool directly.

## Request signing

Requests are signed with HMAC-SHA256 over `METHOD\nPATH\nTIMESTAMP\nNONCE\nSHA256(body)`,
hex-encoded in upper case, and sent as `X-Service-Id`, `X-Timestamp`, `X-Nonce` and
`X-Signature`. The vector asserted in `api-client` is mirrored by
`ServiceSignatureVectorTests` in the .NET test suite, so the two implementations cannot drift
apart silently.

## Not implemented yet

- **Upstream connection sharing.** Each miner gets its own pool connection. Multiplexing several
  miners over one upstream connection is an optimisation, not a correctness requirement.
- **TLS to the pool.** Pools that require `stratum+ssl` need a TLS wrapper around the upstream
  socket.
- **Share value.** The proxy reports proof-of-work detail, not a coin amount, so `coinValue`
  stays null until the pool reports it elsewhere.
