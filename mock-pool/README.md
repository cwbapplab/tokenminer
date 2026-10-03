# Mock mining pool

A standalone Stratum TCP pool that stands in for a real one (Kryptex, or anything the stratum
proxy relays to) while developing and testing. It serves **Pearl (PRL)** and **Quantus (QTC)** on
separate listeners and replies with a **mocked verdict for every share-validation outcome**.

The catch is that these coins do **not** speak classic Bitcoin Stratum V1. Each listener speaks its
coin's own dialect:

| Coin | Dialect | Params | Handshake | `notify` payload |
|---|---|---|---|---|
| Pearl (`prl`) | Pearl named-params (XMRig-style) | object | `mining.authorize` only, no subscribe | `{header, height, job_id, target}` — 76-byte header |
| Quantus (`qtc`) | Classic Stratum V1 (placeholder) | array | `mining.subscribe` + `mining.authorize` | 9-field Bitcoin-style array |

Pearl's dialect follows the published [suprnova Pearl stratum spec](https://prl.suprnova.cc/stratum-spec.html)
v1.4. Quantus's pool-side dialect is not publicly specified — its upstream protocol is QUIC, not
Stratum — so the QTC listener keeps classic V1 until a spec or a captured handshake pins it down.

```
miner ──▶ nginx ──▶ stratum proxy ──▶ mock pool
```

It never talks to the TokenMiner API — it is only the upstream pool.

## What it does

1. Listens on one TCP port per coin, newline-delimited JSON-RPC, LF framing (a trailing CR is
   tolerated on read).
2. Speaks each coin's handshake:
   - **Pearl**: `params` is always an **object**; the client authorizes immediately
     (`{wallet, worker, pass, agent}`) with **no `mining.subscribe`**. The pool pushes the first
     `mining.notify` **before** the authorize ack, per spec §4.1. Errors are objects,
     `{"code":…,"msg":…}`. `mining.submit` is `{job_id, plain_proof}`.
   - **Classic**: `mining.subscribe`, `mining.authorize`, `mining.configure`,
     `mining.suggest_difficulty`, `mining.extranonce.subscribe`, `mining.ping`; array params;
     `[code,msg,traceback]` errors.
3. Hands out work as soon as it is asked for it — for classic, on `subscribe` or on a successful
   `authorize`, whichever comes first; for Pearl, on `authorize`. The previous job is superseded, so
   a late submission against it is reported stale.
4. Behaves like a live pool while connected: on every interval it pushes a `mining.notify` for the
   next block and retargets the share difficulty (vardiff) toward one accepted share per
   `MOCK_POOL_TARGET_SHARE_SECONDS`.
5. Resolves the verdict for each `mining.submit` from the submission itself — no control channel, no
   state to reset between runs — and replies with the matching accept or error.
6. Logs every job push and every verdict with coin, worker/wallet, job and tag.

## Coins and ports

| Coin | Code | Protocol | Listener (default) |
|---|---|---|---|
| Pearl | `prl` | Pearl named-params | `0.0.0.0:3335` |
| Quantus | `qtc` | Classic V1 | `0.0.0.0:3336` |

Pearl job ids look like `<8 lowercase hex>_<int>` (e.g. `18e4eadf_1`); classic ids carry the ticker
(`QTC-1`). To exercise the mock through the proxy, register a pool whose `stratumEndpoint` points at
`<host>:3335` (PRL) or `<host>:3336` (QTC).

## Choosing the verdict

There is no HTTP surface: the verdict is encoded in the submission. A two-character **tag** decides
it — the first two characters of the **nonce** (classic) or of the **`plain_proof`** (Pearl) — and the
**job id** is the fallback. Replay the same submit and you get the same verdict.

| Tag | Outcome | Classic reply | Pearl reply |
|---|---|---|---|
| `00` | accepted | `{"result":true,"error":null}` | `{"result":true,"error":null}` |
| `ff` | bare rejection | `{"result":false,"error":null}` | `{"result":false,"error":null}` |
| `20` | other | `error:[20,"Other/unknown",null]` | `error:{"code":20,"msg":"method not supported"}` |
| `21` | stale job | `error:[21,"Job not found",null]` | `error:{"code":21,"msg":"stale job"}` |
| `22` | duplicate | `error:[22,"Duplicate share",null]` | `error:{"code":22,"msg":"duplicate share"}` |
| `23` | low difficulty | `error:[23,"Low difficulty share",null]` | `error:{"code":23,"msg":"low difficulty share"}` |
| `24` | unauthorized | `error:[24,"Unauthorized worker",null]` | — |
| `25` | not subscribed | `error:[25,"Not subscribed",null]` | — |
| `26` | invalid proof | `error:[20,"Invalid proof",null]` | `error:{"code":26,"msg":"invalid proof"}` |
| anything else | accepted | `{"result":true,"error":null}` | `{"result":true,"error":null}` |

With no tag, a submission is stale when its job id has been superseded by a newer block, or when the
id starts with `stale` or `invalid`. A tag always wins over the job id.

A submission the tag/job rules would **accept** is remembered per connection as `(job_id, tag)`, so
re-submitting the same work returns `22 duplicate share` — a hammering client cannot inflate the
accept count with an identical proof.

### Protocol errors the pool raises on its own

Pearl:
- `mining.authorize` without a `wallet` → `{"code":24,"msg":"wallet is missing"}`.
- `mining.authorize` whose wallet ends in `unauthorized` → `{"code":25,"msg":"invalid wallet"}`.
- `mining.submit` before `mining.authorize` → `{"code":27,"msg":"unauthorized"}`.
- `mining.submit` missing `job_id` or `plain_proof` → `{"code":26,"msg":"invalid proof"}`.
- Any method other than authorize/submit (e.g. `mining.subscribe`) → `{"code":20,"msg":"method not supported"}`.

Classic:
- `mining.submit` before any work has been handed out → `25 not subscribed`.
- `mining.submit` before `mining.authorize` → `24 unauthorized worker`.
- `mining.submit` whose worker differs from the authorized username → `24 unauthorized worker`.
- `mining.authorize` with an empty username → `24 unauthorized worker`.
- A submit missing the worker, job id or nonce → `20 invalid params`.

Both: unparseable JSON → a `20 parse error` reply (the connection survives).

## Configuration

All configuration comes from the environment.

| Variable | Default | Meaning |
|---|---|---|
| `MOCK_POOL_PRL_ADDR` | `0.0.0.0:3335` | Pearl listener address |
| `MOCK_POOL_QTC_ADDR` | `0.0.0.0:3336` | Quantus listener address |
| `MOCK_POOL_DIFFICULTY` | `2.0` | Difficulty sent in classic `mining.set_difficulty` |
| `MOCK_POOL_PEARL_DIFFICULTY` | `1000000000` | Pearl starting share difficulty; the pushed `target` is `2^256 / difficulty` |
| `MOCK_POOL_TARGET_SHARE_SECONDS` | `10` | Vardiff target: aim for one accepted share per this many seconds; `0` disables vardiff |
| `MOCK_POOL_JOB_INTERVAL_SECONDS` | `5` | Seconds between `mining.notify` pushes; `0` disables them |
| `MOCK_POOL_LOG_LEVEL` | `info` | `debug`, `info`, `warning` or `error` |

## Running

```bash
dotnet run --project mock-pool/src/MockPool
```

Pearl (note the job arrives before the ack; a `23…` proof asks for a low-difficulty rejection):

```
{"id":1,"method":"mining.authorize","params":{"wallet":"prl1p…","worker":"rig1","pass":"x","agent":"peakminer/2.17.5"}}
{"id":null,"method":"mining.notify","params":{"header":"00004020…ffff001d","height":100001,"job_id":"18e4eadf_1","target":"000000044b…"}}
{"id":1,"result":true,"error":null}
{"id":10,"method":"mining.submit","params":{"job_id":"18e4eadf_1","plain_proof":"23AAAA…"}}
{"id":10,"result":null,"error":{"code":23,"msg":"low difficulty share"}}
```

Classic:

```
{"id":1,"method":"mining.subscribe","params":["mock-miner/1.0"]}
{"id":1,"result":[[["mining.set_difficulty","…"],["mining.notify","…"]],"a1b2c3d4",4],"error":null}
{"id":null,"method":"mining.set_difficulty","params":[2.0]}
{"id":null,"method":"mining.notify","params":["QTC-1","…","…","…",[],"20000000","1d00ffff","…",true]}
{"id":2,"method":"mining.authorize","params":["wallet.rig-1","x"]}
{"id":2,"result":true,"error":null}
{"id":3,"method":"mining.submit","params":["wallet.rig-1","QTC-1","deadbeef","65a1b2c3","23aabbcc"]}
{"id":3,"result":null,"error":[23,"Low difficulty share",null]}
```

## Testing

```bash
dotnet test mock-pool/tests/MockPool.Tests
```

The suite covers the tag/job verdict table, the wire envelopes for both dialects, and full socket
handshakes for Pearl and classic.

## Container

```bash
docker build -t tokenminer-mock-pool ./mock-pool
docker run --rm -p 3335:3335 -p 3336:3336 tokenminer-mock-pool
```

## Not implemented

- **Quantus's pool dialect.** The QTC listener runs classic V1 as a placeholder; Quantus's real
  upstream is a QUIC protocol and no pool-side spec exists. Capture `peakminer --coin quantus`
  against the logging forwarder to pin it down, then add a `CoinProtocol.Quantus` dialect.
- **No proof verification.** Pearl's `plain_proof` is treated as an opaque string; the pool never
  checks the PlainProof, Merkle proofs or jackpot hash. Only the tag/job convention drives the
  verdict, plus duplicate detection on `(job_id, tag)`.
- **No chain behind the jobs.** New jobs advance the block on a timer; there is no real block, merkle
  root or difficulty retarget behind them.
- **No TLS.** Points of the pool that require `stratum+ssl` need a TLS wrapper.
