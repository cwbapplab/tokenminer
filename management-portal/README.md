# TokenMiner management portal

Operator console for the data the API owns: the mining catalogue, the treasury pipeline and the
LLM-provider accounts. Built with React + TypeScript on Vite, styled after the
[Jobick](https://jobick.dexignlab.com/xhtml/index.html) admin template.

## Running

The portal talks to the API on port 5210 and Vite proxies `/api` and `/health` to it, so no CORS
configuration is needed on the backend.

```powershell
# 1. backend (from the repository root)
./scripts/dev.ps1 up

# 2. portal
cd management-portal
npm install
npm run dev            # http://localhost:5273
```

Point it at a different API with `VITE_API_URL`, e.g. `$env:VITE_API_URL='http://localhost:5211'`.

```powershell
npm run build          # typecheck + production bundle into dist/
npm run typecheck      # tsc --noEmit
```

## Signing in

Every screen requires the `admin` role; a session without it is told so rather than shown an
empty console.

To create the first administrator:

1. Add the address to `Auth:BootstrapAdminEmails` and start the API. On boot it grants the
   `admin` role to any account with that address.
2. Register the account (`POST /api/auth/register`) — or use the API's Swagger page.
3. Sign in here. The sign-in form recognises an account that still needs activating, sends the
   one-time code and activates it, so the portal does not depend on any other client.

With SMTP unconfigured the development email sender writes the verification token and the
activation code to the API log, which is the easiest place to read them.

## Screens

| Screen | What it does |
|---|---|
| Dashboard | Live status, daily payout chart, conversions by state, recent activity. |
| Users | Accounts with their client id, device count and accepted/rejected share counts. Select a user to see the hardware in use, or a device count to open the user's hardware list. |
| Coins | Mined coins and stablecoins, with their last known price. |
| Pools | Pools, their payout rail, mode and coin set. |
| Algorithms | Miner launch configurations, priority and coin matching. |
| Conversion providers | Swap services; credentials are references, never secrets. |
| Conversion routes | Which provider converts which pair, and the minimum amount. |
| Conversions | The swap ledger, with manual settlement for off-system swaps. |
| Pool payouts | What each pool has paid out and how much is attributed to shares. |
| LLM providers | The gateways topped up, with deposit rails and the synced model catalogue. |
| Provider deposits | Stablecoin transfers, and whether the provider has credited them. |
| Configuration | Runtime settings that override the deployed defaults. |

## Adding a screen

Most screens are generated from a schema. To add one, describe it in
`src/lib/resources.ts` — endpoint, columns, form fields — and it appears in the sidebar:

```ts
const widgets: Resource = {
  key: 'widgets',
  path: '/api/admin/widgets',
  title: 'Widgets',
  singular: 'widget',
  description: '…',
  category: 'Mining',
  canCreate: true,
  canEdit: true,
  columns: [{ key: 'name', label: 'Name', format: 'strong' }],
  fields: [{ name: 'name', label: 'Name', type: 'text', required: true }],
};
```

Field schemas handle nested objects (`nested: 'details'`), multi-selects
(`type: 'multiselect'`, `optionsFrom: 'coins'`), JSON payloads (`type: 'json'`) and which fields
apply to create versus edit (`createOnly` / `editOnly`). The form engine builds the payload,
including folding nested fields into their container.

## Notes and limits

- **The ledgers are read-only on purpose.** Payouts, deposits and conversions are produced by the
  background pipeline; the only manual action is settling a conversion.
- **The user, hardware and share views are read-only too.** They are driven by
  `GET /api/admin/users` and `GET /api/admin/users/{id}`, which surface the account, its devices and
  the terminal (accepted/redeemed and rejected) share counts per device. In-flight shares are not
  counted; there is no admin endpoint for `mining_statistics` yet.
- **The bundle is one chunk** (~375 KB gzipped), dominated by the charting library. Fine for an
  internal console; split it if the portal ever becomes public.
- **Poppins is fetched from Google Fonts** at runtime and falls back to the system stack offline.
