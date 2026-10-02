# TokenMiner backend

Turns user mining work into LLM-provider credit.

```
miner ──▶ nginx ──▶ stratum proxy ──▶ mining pool
                         │
                         │ accepted shares only
                         ▼
                    .NET API ──▶ shares ──▶ payouts ──▶ conversion ──▶ provider deposit
                                                                          │
                                                                          ▼
                                                                    router.one USD
```

The API never sits in the Stratum hot path. Mining, rewards, treasury and provider billing are
independent domains that communicate only through durable financial records, and every event is
independently retryable and idempotent.

## Projects

| Project | Contents |
|---|---|
| `TokenMiner.Domain` | Entities and enums. No dependencies, no persistence attributes. |
| `TokenMiner.Application` | Use cases (MediatR), ports, validators. No EF, no HTTP. |
| `TokenMiner.Contracts` | Wire DTOs, shared with the client and the Rust proxy. |
| `TokenMiner.Infrastructure` | EF Core, integrations, background jobs, secrets. |
| `TokenMiner.Api` | ASP.NET Core host: endpoints, WebSocket, auth wiring. |

Related: [`../stratum-proxy`](../stratum-proxy) (Rust) and [`../deploy/nginx`](../deploy/nginx).

## Running

```bash
docker compose up -d postgres          # PostgreSQL on 5433
dotnet run --project backend/src/TokenMiner.Api
```

Swagger is at `/swagger` in Development. Integration tests need Docker because they start a real
PostgreSQL container.

```bash
dotnet test                            # 246 tests
cd ../stratum-proxy && cargo test      # 37 tests
```

## Endpoints

| Area | Routes |
|---|---|
| Auth | `POST /api/auth/{register,verify-email,otp/request,otp/verify,google,login,refresh,logout}`, `GET /api/auth/me` |
| Mining | `POST /api/mining/{start,stop}`, `WS /ws/mining`, `GET /api/analytics` |
| Internal (signed) | `POST /internal/mining/shares`, `GET /internal/stratum/{config,workers/{id}}` |
| Admin: catalogue | `/api/admin/{coins,pools,mining-algos}` |
| Admin: treasury | `/api/admin/{conversion-providers,conversion-routes,conversions,pool-payouts}` |
| Admin: providers | `/api/admin/{llm-providers,provider-deposits}` |
| Admin: system | `/api/admin/system/{status,configurations}` |

Admin routes require the `admin` role. Grant it by registering an account, setting
`Auth:BootstrapAdminEmails`, and restarting.

## Two authentication schemes

- **Users** present a JWT access token (short-lived) with a rotating refresh token. The mining
  WebSocket also accepts the token on the query string, because a browser-based client cannot set
  headers on a WebSocket.
- **Services** (the stratum proxy) sign each request: HMAC-SHA256 over
  `METHOD\nPATH\nTIMESTAMP\nNONCE\nSHA256(body)`, hex-encoded upper case, sent as
  `X-Service-Id`, `X-Timestamp`, `X-Nonce` and `X-Signature`. Verification runs in middleware
  before endpoint binding — minimal APIs bind arguments before filters run, so a filter would see
  an already-consumed body. Used nonces are stored so a replay is rejected.

## Idempotency

Every financial step has a key, so a retry cannot double-spend:

| Step | Key |
|---|---|
| Share | `(pool_id, share_identifier)` |
| Pool payout | `(pool_id, transaction_hash)` + `redeemed_amount` |
| Conversion | `payout:{payoutId}` |
| Provider deposit | `conversion:{conversionId}` |

## Configuration

Settings bind from configuration in this order of precedence: `system_configurations` (runtime,
operator-editable) → `appsettings.json` → code defaults.

| Section | Purpose |
|---|---|
| `Auth` | JWT signing, OTP policy, lockout, rate limits, bootstrap admins |
| `Mining` | Heartbeat grace, watchdog and job cadences, service-auth secret, public stratum endpoint |
| `Treasury` | Pool monitor, coin prices, payout confirmation threshold, conversion adapters |
| `Providers` | Balance/model/deposit cadences, confirmation threshold, top-up thresholds, simulated chain |

Secrets are never stored in a database column: rows hold a `credential_ref`, resolved through
`ISecretResolver`. The shipped resolver reads `Secrets:<ref>` from configuration; a vault adapter
replaces it without touching callers.

## Background services

Each is a `PeriodicJob` — its own dependency-injection scope, and a failing tick is logged and
swallowed so one bad run cannot kill the loop. They are hosted services because the set is small;
the plan's durable job runner is the natural next step for the money-moving ones.

`MiningSessionWatchdog` · `ShareRewardProcessorJob` · `MiningStatisticsRollupJob` · `PoolMonitorJob` ·
`PayoutReconciliationJob` · `CoinPriceJob` · `ConversionJob` · `ProviderBalanceJob` ·
`ModelCatalogSyncJob` · `ProviderDepositPipelineJob` · `AutoTopUpJob`

## External integrations

- **Kryptex** — balance and payout history over its public API; the withdrawal *request* is web-only,
  so that one action is isolated behind `IPoolWithdrawalClient` with a Playwright implementation that
  is **off by default**.
- **router.one** — `GET /v1/balance` for reconciliation and `GET /v1/models` for the catalogue. There
  is no documented top-up endpoint, so deposits are sent on chain and only marked `credited` when the
  balance actually grows.
- **Blockchain** — `IBlockchainTransferProvider` with a simulated implementation. A live EVM adapter
  needs a funded hot wallet and a key-custody decision, which is why the port exists.

## Known gaps

Honest list of what is *not* done, so nothing here is mistaken for finished:

1. **No live exchange adapter.** `IConversionProvider` has manual and simulated implementations; the
   deposit/withdrawal/balance/status operations a real exchange needs are not declared, because
   nothing would implement them yet.
2. **No live blockchain adapter.** Transfers are simulated. Sending real stablecoins needs a funded
   wallet and key custody.
3. **The Playwright withdrawal path is unverified.** It is off by default and needs
   `playwright install chromium` plus selectors confirmed against the live page.
4. **The container image for the stratum proxy was never built.** The exact build command it uses was
   verified, and the resulting binary was smoke-tested, but `docker build` itself was not run.
5. **nginx was never exercised.** The `stream` passthrough is configured but no traffic was routed
   through it.
6. **No load or penetration testing.** The rate limits, lockout and service auth have unit and
   integration coverage; nothing has been stress-tested.
7. **`mining_statistics` is rebuilt by loading every rewardable share.** Fine at current volume,
   wrong at scale — it needs windowed aggregation.
8. **One upstream pool connection per miner.** Correct but not efficient; multiplexing is an
   optimisation.
9. **Google sign-in is untested end to end** — it needs a real Google client id, so only the token
   validation path has unit coverage.
