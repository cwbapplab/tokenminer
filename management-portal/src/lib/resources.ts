import type { ReactNode } from 'react';
import { coinCode } from './lookups';
import type { Lookups } from './lookups';
import {
  formatDate,
  formatDateTime,
  formatInt,
  formatNumber,
  formatRelative,
  formatUsd,
  humanize,
  truncateMiddle,
} from './format';

/** Rows come straight off the API, so they are untyped at this layer by design. */
export type Row = Record<string, unknown>;

export type ColumnFormat =
  | 'text'
  | 'strong'
  | 'badge'
  | 'datetime'
  | 'date'
  | 'relative'
  | 'usd'
  | 'number'
  | 'int'
  | 'mono'
  | 'bool'
  | 'coinCodes';

export interface Column {
  key: string;
  label: string;
  format?: ColumnFormat;
  /** Escape hatch for columns that need the reference data, e.g. showing a coin code. */
  render?: (row: Row, lookups: Lookups) => ReactNode;
  align?: 'end';
  hideOnNarrow?: boolean;
}

export type FieldType =
  | 'text'
  | 'number'
  | 'decimal'
  | 'select'
  | 'multiselect'
  | 'checkbox'
  | 'textarea'
  | 'json';

export interface SelectOption {
  value: string;
  label: string;
}

export interface Field {
  name: string;
  label: string;
  type: FieldType;
  required?: boolean;
  placeholder?: string;
  hint?: string;
  options?: SelectOption[];
  /** Options supplied at render time from data the API owns. */
  optionsFrom?: 'coins' | 'conversionProviders' | 'llmProviders';
  /** Writes the value into a nested object on the payload, e.g. `details`. */
  nested?: string;
  createOnly?: boolean;
  editOnly?: boolean;
  defaultValue?: unknown;
  fullWidth?: boolean;
}

export type ActionKind = 'settle' | 'depositAccounts' | 'models';

export interface ResourceAction {
  kind: ActionKind;
  label: string;
}

export interface Resource {
  key: string;
  path: string;
  title: string;
  singular: string;
  description: string;
  category: 'Mining' | 'Treasury' | 'Providers';
  columns: Column[];
  fields: Field[];
  canCreate: boolean;
  canEdit: boolean;
  /** Overrides the PUT target, e.g. `/api/admin/conversion-routes/{id}`. */
  updatePath?: (row: Row) => string;
  actions?: ResourceAction[];
}

const STATUS_OPTIONS: SelectOption[] = [
  { value: 'active', label: 'Active' },
  { value: 'disabled', label: 'Disabled' },
];

const PAYOUT_MODE_OPTIONS: SelectOption[] = [
  { value: 'managed_request', label: 'Managed request (we ask the pool to pay out)' },
  { value: 'native_auto_payout', label: 'Native auto payout (the pool decides)' },
  { value: 'direct_to_wallet', label: 'Direct to wallet (we mine to our own wallet)' },
];

function statusColumn(key = 'status'): Column {
  return { key, label: 'Status', format: 'badge' };
}

// --- Mining ---------------------------------------------------------------------------------

const coins: Resource = {
  key: 'coins',
  path: '/api/admin/coins',
  title: 'Coins',
  singular: 'coin',
  description: 'Mined coins and the stablecoins they convert into, with their last known price.',
  category: 'Mining',
  canCreate: true,
  canEdit: true,
  columns: [
    { key: 'code', label: 'Code', format: 'strong' },
    { key: 'name', label: 'Name' },
    { key: 'network', label: 'Network' },
    { key: 'decimals', label: 'Decimals', format: 'int', align: 'end' },
    {
      key: 'lastKnownUsdValue',
      label: 'Price',
      format: 'usd',
      align: 'end',
      render: (row) => (row.lastKnownUsdValue === null ? '—' : formatUsd(row.lastKnownUsdValue as number, 6)),
    },
    { key: 'lastValueDate', label: 'Priced', format: 'relative', hideOnNarrow: true },
    statusColumn(),
  ],
  fields: [
    { name: 'code', label: 'Code', type: 'text', required: true, createOnly: true, placeholder: 'prl', hint: 'Must match the pool’s own coin slug.' },
    { name: 'name', label: 'Name', type: 'text', required: true, placeholder: 'Pearl' },
    { name: 'network', label: 'Network', type: 'text', required: true, placeholder: 'bep20' },
    { name: 'decimals', label: 'Decimals', type: 'number', required: true, defaultValue: 8 },
    { name: 'status', label: 'Status', type: 'select', options: STATUS_OPTIONS, editOnly: true },
  ],
};

const pools: Resource = {
  key: 'pools',
  path: '/api/admin/pools',
  title: 'Pools',
  singular: 'pool',
  description: 'Mining pools, their payout rail and the coins each one mines.',
  category: 'Mining',
  canCreate: true,
  canEdit: true,
  columns: [
    { key: 'name', label: 'Pool', format: 'strong' },
    { key: 'systemPoolId', label: 'System id', format: 'mono' },
    { key: 'provider', label: 'Payout provider' },
    { key: 'details.payoutMode', label: 'Payout mode', render: (row) => humanize((row.details as Row | null)?.payoutMode as string) },
    { key: 'details.payoutAddress', label: 'Payout address', format: 'mono', hideOnNarrow: true },
    {
      key: 'coinIds',
      label: 'Coins',
      render: (row, lookups) => {
        const ids = (row.coinIds as string[]) ?? [];
        return ids.length === 0 ? '—' : ids.map((id) => coinCode(lookups.coins, id)).join(', ');
      },
    },
    statusColumn(),
  ],
  fields: [
    { name: 'systemPoolId', label: 'System pool id', type: 'text', required: true, createOnly: true, placeholder: 'kryptex-qtc' },
    { name: 'name', label: 'Name', type: 'text', required: true, placeholder: 'Kryptex QTC' },
    { name: 'provider', label: 'Payout provider', type: 'text', required: true, defaultValue: 'kryptex', hint: 'Matches the payout integration, e.g. kryptex.' },
    { name: 'status', label: 'Status', type: 'select', options: STATUS_OPTIONS, editOnly: true },

    { name: 'baseUrl', label: 'Pool base URL', type: 'text', required: true, nested: 'details', placeholder: 'https://pool.kryptex.com' },
    { name: 'statusEndpoint', label: 'Status endpoint', type: 'text', nested: 'details', hint: 'Optional monitoring endpoint.' },
    { name: 'stratumEndpoint', label: 'Stratum endpoint', type: 'text', nested: 'details', placeholder: 'qtc-br.kryptex.network:7777', hint: 'Where the proxy relays to.' },
    { name: 'coinId', label: 'Payout coin', type: 'select', required: true, nested: 'details', optionsFrom: 'coins', hint: 'The coin this pool actually pays out.' },
    { name: 'payoutMode', label: 'Payout mode', type: 'select', required: true, nested: 'details', options: PAYOUT_MODE_OPTIONS, defaultValue: 'native_auto_payout' },
    { name: 'payoutAddress', label: 'Payout address', type: 'text', nested: 'details' },
    { name: 'payoutNetwork', label: 'Payout network', type: 'text', nested: 'details', placeholder: 'bep20' },
    { name: 'coinIds', label: 'Coins mined here', type: 'multiselect', required: true, optionsFrom: 'coins', fullWidth: true },
  ],
};

const miningAlgos: Resource = {
  key: 'mining-algos',
  path: '/api/admin/mining-algos',
  title: 'Mining algorithms',
  singular: 'algorithm',
  description: 'Miner launch configurations. The highest-priority active algorithm whose coin list matches wins.',
  category: 'Mining',
  canCreate: true,
  canEdit: true,
  columns: [
    { key: 'code', label: 'Code', format: 'strong' },
    { key: 'name', label: 'Name' },
    { key: 'priority', label: 'Priority', format: 'int', align: 'end' },
    {
      key: 'configuration',
      label: 'Command',
      format: 'mono',
      render: (row) => {
        const config = row.configuration as { command?: string } | null;
        return config?.command ? truncateMiddle(config.command, 34, 10) : '—';
      },
    },
    {
      key: 'configuration.coins',
      label: 'Coins',
      render: (row) => {
        const config = row.configuration as { supportedCoins?: string[] } | null;
        const supported = config?.supportedCoins ?? [];
        return supported.length === 0 ? 'any' : supported.join(', ');
      },
    },
    statusColumn(),
  ],
  fields: [
    { name: 'code', label: 'Code', type: 'text', required: true, createOnly: true, placeholder: 'rgminer-prl' },
    { name: 'name', label: 'Name', type: 'text', required: true, placeholder: 'RG Miner (PRL)' },
    { name: 'priority', label: 'Priority', type: 'number', required: true, defaultValue: 10, hint: 'Higher wins when several match.' },
    { name: 'status', label: 'Status', type: 'select', options: STATUS_OPTIONS, editOnly: true },
    {
      name: 'configuration',
      label: 'Configuration (JSON)',
      type: 'json',
      fullWidth: true,
      hint: 'e.g. { "command": "rgminer.exe -a {algo} -o {poolUrl} -u {workerId}", "supportedCoins": ["PRL"] }',
      defaultValue: { command: '', supportedCoins: [] },
    },
  ],
};

// --- Treasury -------------------------------------------------------------------------------

const conversionProviders: Resource = {
  key: 'conversion-providers',
  path: '/api/admin/conversion-providers',
  title: 'Conversion providers',
  singular: 'conversion provider',
  description: 'Swap services that turn a mined coin into a stablecoin. Credentials are references, never secrets.',
  category: 'Treasury',
  canCreate: true,
  canEdit: true,
  columns: [
    { key: 'name', label: 'Name', format: 'strong' },
    { key: 'baseUrl', label: 'Base URL', format: 'mono' },
    { key: 'priority', label: 'Priority', format: 'int', align: 'end' },
    { key: 'credentialRef', label: 'Credential ref', format: 'mono', hideOnNarrow: true },
    statusColumn(),
  ],
  fields: [
    { name: 'name', label: 'Name', type: 'text', required: true, createOnly: true, hint: 'Must match a registered adapter, e.g. simulated or manual.' },
    { name: 'baseUrl', label: 'Base URL', type: 'text', required: true, placeholder: 'https://example.invalid' },
    { name: 'priority', label: 'Priority', type: 'number', required: true, defaultValue: 10 },
    { name: 'credentialRef', label: 'Credential reference', type: 'text', hint: 'Key into the secret resolver.' },
    { name: 'supportedFeatures', label: 'Supported features', type: 'textarea', fullWidth: true },
    { name: 'status', label: 'Status', type: 'select', options: STATUS_OPTIONS, editOnly: true },
  ],
};

const conversionRoutes: Resource = {
  key: 'conversion-routes',
  path: '/api/admin/conversion-routes',
  title: 'Conversion routes',
  singular: 'route',
  description: 'Which provider converts which pair, and the minimum amount worth converting.',
  category: 'Treasury',
  canCreate: true,
  canEdit: true,
  updatePath: (row) => `/api/admin/conversion-routes/${String(row.id)}`,
  columns: [
    { key: 'sourceCoinCode', label: 'From', format: 'strong' },
    { key: 'destinationCoinCode', label: 'To', format: 'strong' },
    { key: 'conversionProviderName', label: 'Provider' },
    { key: 'priority', label: 'Priority', format: 'int', align: 'end' },
    { key: 'minimumAmount', label: 'Minimum', format: 'number', align: 'end' },
    {
      key: 'enabled',
      label: 'Enabled',
      render: (row) => (row.enabled ? 'Yes' : 'No'),
    },
  ],
  fields: [
    { name: 'conversionProviderId', label: 'Provider', type: 'select', required: true, createOnly: true, optionsFrom: 'conversionProviders' },
    { name: 'sourceCoinId', label: 'Source coin', type: 'select', required: true, createOnly: true, optionsFrom: 'coins' },
    { name: 'destinationCoinId', label: 'Destination coin', type: 'select', required: true, createOnly: true, optionsFrom: 'coins' },
    { name: 'sourceNetwork', label: 'Source network', type: 'text', createOnly: true, placeholder: 'bep20' },
    { name: 'destinationNetwork', label: 'Destination network', type: 'text', createOnly: true, placeholder: 'bep20' },
    { name: 'priority', label: 'Priority', type: 'number', required: true, defaultValue: 10 },
    { name: 'minimumAmount', label: 'Minimum amount', type: 'decimal', required: true, defaultValue: 0 },
    { name: 'enabled', label: 'Enabled', type: 'checkbox', editOnly: true, defaultValue: true },
  ],
};

// --- Providers ------------------------------------------------------------------------------

const llmProviders: Resource = {
  key: 'llm-providers',
  path: '/api/admin/llm-providers',
  title: 'LLM providers',
  singular: 'LLM provider',
  description: 'The prepaid gateways the mined value is converted into credit at.',
  category: 'Providers',
  canCreate: true,
  canEdit: true,
  actions: [{ kind: 'depositAccounts', label: 'Deposit rails' }, { kind: 'models', label: 'Models' }],
  columns: [
    { key: 'name', label: 'Provider', format: 'strong' },
    { key: 'endpoint', label: 'Endpoint', format: 'mono' },
    {
      key: 'balance',
      label: 'Balance',
      align: 'end',
      render: (row) => formatUsd(row.balance as number | null),
    },
    {
      key: 'reservedBalance',
      label: 'Reserved',
      align: 'end',
      hideOnNarrow: true,
      render: (row) => formatUsd(row.reservedBalance as number | null),
    },
    {
      key: 'depositAccounts',
      label: 'Rails',
      align: 'end',
      render: (row) => ((row.depositAccounts as unknown[] | undefined)?.length ?? 0).toString(),
    },
    { key: 'balanceCheckedAt', label: 'Checked', format: 'relative', hideOnNarrow: true },
    statusColumn(),
  ],
  fields: [
    { name: 'name', label: 'Name', type: 'text', required: true, hint: 'Must match a registered client, e.g. router-one.' },
    { name: 'endpoint', label: 'Endpoint', type: 'text', required: true, placeholder: 'https://api.router.one/v1' },
    { name: 'credentialRef', label: 'Credential reference', type: 'text', hint: 'Resolved through the secret resolver, never stored here.' },
    { name: 'status', label: 'Status', type: 'select', options: STATUS_OPTIONS, editOnly: true },
  ],
};

export const RESOURCES: Resource[] = [
  coins,
  pools,
  miningAlgos,
  conversionProviders,
  conversionRoutes,
  llmProviders,
];

export const RESOURCE_CATEGORIES: Resource['category'][] = ['Mining', 'Treasury', 'Providers'];

export function findResource(key: string | undefined): Resource | undefined {
  return RESOURCES.find((resource) => resource.key === key);
}

/** Column formatters shared with the read-only ledger tables. */
export const badgeOrDash = (value: unknown): string => humanize(value as string | null);
export const dateOrDash = formatDate;
export const dateTimeOrDash = formatDateTime;
export const usdOrDash = formatUsd;
export const numberOrDash = formatNumber;
export const intOrDash = formatInt;
export const relativeOrDash = formatRelative;
