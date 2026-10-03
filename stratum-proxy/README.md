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

## Coin dialects

The relay is verbatim, so it carries any Stratum dialect unchanged; parsing exists only to observe
the worker and the shares. Two dialects are recognised:

- **Classic** (positional): `mining.authorize ["wallet.worker","x"]` and
  `mining.submit [worker, jobId, extranonce2, ntime, nonce]`.
- **Pearl** (named params, no `mining.subscribe`): `mining.authorize {"wallet":…,"worker":…}`
  and `mining.submit {"job_id":…,"plain_proof":…}`.

A share's idempotency key is derived from the work — the nonce for classic, the `plain_proof` for
Pearl — so re-submitting the same work collapses to one record.

## Routing

The username carries everything the proxy needs, so it never calls the API before dialing a pool:

```
wallet[.systemPoolId].worker      worker = <userId>-<hardwareId> (two canonical GUIDs)
```

- **Pool** — if a `systemPoolId` segment is present it must name a pool in the routing table; if it
  is omitted the pool is inferred from the wallet (which must then be unique to one pool).
- **Worker** — always the last dot-segment, split at the fixed GUID boundary into the user id and
  the hardware id used to attribute the share.
- **Endpoint** — taken from the routing table entry, in memory.

## Wallet policy

At startup the proxy loads the active pools and keeps a `coin -> wallet` map in memory (the wallet
each active pool pays out to), refreshed with the routing table. On `mining.authorize`:

1. The wallet — everything before the first dot of the username — must belong to one of those
   pools, or the connection is refused from memory with `24` and never reaches a pool.
2. Once the worker is resolved and its coin is known, the wallet must be exactly that coin's active
   pool wallet; a mismatch is refused the same way.

So a miner has to mine with the pool's own wallet; a wallet that would redirect payouts elsewhere
cannot start a session. (The gate is skipped while no pool has a payout wallet configured, so an
empty catalogue does not lock everyone out.)

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
