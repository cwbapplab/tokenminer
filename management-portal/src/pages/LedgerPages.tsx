import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { api } from '../lib/api';
import { useLookups } from '../lib/lookups';
import { coinCode } from '../lib/lookups';
import type { Lookups } from '../lib/lookups';
import { formatDateTime, formatNumber, formatUsd, truncateMiddle } from '../lib/format';
import type { Column, Row } from '../lib/resources';
import { Icon } from '../components/icons';
import { DataTable, Modal, useAsyncList, useToast } from '../components/ui';
import type { ConversionTransaction, PoolPayout, ProviderDeposit } from '../lib/types';

const LIMITS = [50, 100, 200, 500];

/** Shared chrome: a title, a limit selector, refresh and the table. */
function LedgerCard({
  title,
  description,
  limit,
  onLimitChange,
  onRefresh,
  children,
}: {
  title: string;
  description: string;
  limit: number;
  onLimitChange: (limit: number) => void;
  onRefresh: () => void;
  children: ReactNode;
}) {
  return (
    <div className="card">
      <div className="card-header">
        <div>
          <h4 className="card-title">{title}</h4>
          <div className="field-hint">{description}</div>
        </div>
        <div className="d-flex align-items-center gap-2">
          <select
            className="form-select"
            style={{ width: '8rem' }}
            value={limit}
            onChange={(event) => onLimitChange(Number(event.target.value))}
            aria-label="Rows to load"
          >
            {LIMITS.map((value) => (
              <option key={value} value={value}>
                Last {value}
              </option>
            ))}
          </select>
          <button type="button" className="btn btn-outline-secondary btn-square" onClick={onRefresh} aria-label="Refresh" title="Refresh">
            <Icon name="refresh" />
          </button>
        </div>
      </div>
      <div className="card-body pt-0">{children}</div>
    </div>
  );
}

// --- Conversions -----------------------------------------------------------------------------

function SettleModal({
  conversion,
  onClose,
  onSaved,
}: {
  conversion: ConversionTransaction;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [outcome, setOutcome] = useState('completed');
  const [destinationAmount, setDestinationAmount] = useState(String(conversion.destinationAmount ?? ''));
  const [exchangeRate, setExchangeRate] = useState(String(conversion.exchangeRate ?? ''));
  const [fees, setFees] = useState(String(conversion.fees ?? ''));
  const [destinationTransactionId, setDestinationTransactionId] = useState(conversion.destinationTransactionId ?? '');
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const { push } = useToast();

  async function submit() {
    setBusy(true);
    setFailure(null);

    try {
      await api.post(`/api/admin/conversions/${conversion.id}/settlement`, {
        outcome,
        destinationAmount: destinationAmount === '' ? null : Number(destinationAmount),
        exchangeRate: exchangeRate === '' ? null : Number(exchangeRate),
        fees: fees === '' ? null : Number(fees),
        destinationTransactionId: destinationTransactionId === '' ? null : destinationTransactionId,
        reason: reason === '' ? null : reason,
      });

      push('Conversion settled.');
      onSaved();
    } catch (caught) {
      setFailure(caught instanceof Error ? caught.message : 'Could not settle the conversion.');
    } finally {
      setBusy(false);
    }
  }

  const completing = outcome === 'completed';

  return (
    <Modal
      title="Settle conversion"
      subtitle={`${conversion.sourceAmount} ${conversion.sourceCoinCode} → ${conversion.destinationCoinCode}`}
      onClose={onClose}
      footer={
        <>
          <button type="button" className="btn btn-outline-secondary" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="button" className="btn btn-primary" onClick={submit} disabled={busy}>
            {busy ? 'Saving…' : 'Record outcome'}
          </button>
        </>
      }
    >
      {failure && <div className="alert alert-danger mb-3">{failure}</div>}

      <div className="alert alert-info mb-4">
        Use this when the swap happened outside the system, or to close out a conversion that the provider
        refused. It records what actually happened; nothing is sent anywhere.
      </div>

      <div className="mb-3">
        <label className="form-label" htmlFor="outcome">
          Outcome
        </label>
        <select className="form-select" id="outcome" value={outcome} onChange={(event) => setOutcome(event.target.value)}>
          <option value="completed">Completed — the funds arrived</option>
          <option value="failed">Failed — the swap did not happen</option>
          <option value="cancelled">Cancelled — abandoned on purpose</option>
        </select>
      </div>

      {completing && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <label className="form-label" htmlFor="destinationAmount">
              Received amount
            </label>
            <input
              className="form-control"
              id="destinationAmount"
              type="number"
              step="any"
              value={destinationAmount}
              onChange={(event) => setDestinationAmount(event.target.value)}
              placeholder={String(conversion.destinationAmount ?? '')}
            />
          </div>
          <div className="col-md-6 mb-3">
            <label className="form-label" htmlFor="exchangeRate">
              Exchange rate
            </label>
            <input
              className="form-control"
              id="exchangeRate"
              type="number"
              step="any"
              value={exchangeRate}
              onChange={(event) => setExchangeRate(event.target.value)}
            />
          </div>
          <div className="col-md-6 mb-3">
            <label className="form-label" htmlFor="fees">
              Fees
            </label>
            <input
              className="form-control"
              id="fees"
              type="number"
              step="any"
              value={fees}
              onChange={(event) => setFees(event.target.value)}
            />
          </div>
          <div className="col-md-6 mb-3">
            <label className="form-label" htmlFor="destinationTransactionId">
              Destination transaction id
            </label>
            <input
              className="form-control"
              id="destinationTransactionId"
              value={destinationTransactionId}
              onChange={(event) => setDestinationTransactionId(event.target.value)}
              placeholder="on-chain hash"
            />
          </div>
        </div>
      )}

      {!completing && (
        <div className="mb-3">
          <label className="form-label" htmlFor="reason">
            Reason
          </label>
          <textarea
            className="form-control"
            id="reason"
            value={reason}
            onChange={(event) => setReason(event.target.value)}
            placeholder="Why this conversion did not go through"
          />
        </div>
      )}
    </Modal>
  );
}

export function ConversionsPage() {
  const [limit, setLimit] = useState(100);
  const conversions = useAsyncList<ConversionTransaction>(`/api/admin/conversions?limit=${limit}`);
  const lookups = useLookups();
  const [settling, setSettling] = useState<ConversionTransaction | null>(null);
  const [onlyPending, setOnlyPending] = useState(false);

  const rows = useMemo<Row[]>(() => {
    const all = conversions.data as unknown as Row[];

    return onlyPending ? all.filter((row) => row.status === 'pending' || row.status === 'processing') : all;
  }, [conversions.data, onlyPending]);

  const columns: Column[] = [
    {
      key: 'source',
      label: 'From',
      format: 'strong',
      render: (row) => `${formatNumber(row.sourceAmount as number)} ${String(row.sourceCoinCode)}`,
    },
    {
      key: 'destination',
      label: 'To',
      render: (row) =>
        row.destinationAmount === null
          ? String(row.destinationCoinCode)
          : `${formatNumber(row.destinationAmount as number)} ${String(row.destinationCoinCode)}`,
    },
    { key: 'conversionProviderName', label: 'Provider' },
    { key: 'exchangeRate', label: 'Rate', align: 'end', render: (row) => formatNumber(row.exchangeRate as number) },
    { key: 'fees', label: 'Fees', align: 'end', render: (row) => formatNumber(row.fees as number) },
    { key: 'status', label: 'Status', format: 'badge' },
    { key: 'idempotencyKey', label: 'Key', format: 'mono', hideOnNarrow: true },
    { key: 'createdAt', label: 'Created', render: (row) => formatDateTime(String(row.createdAt)) },
    {
      key: 'error',
      label: 'Error',
      render: (row) => (row.error ? <span className="fs-13" style={{ color: 'var(--tm-danger)' }}>{String(row.error)}</span> : <span className="text-muted">—</span>),
    },
  ];

  return (
    <>
      <LedgerCard
        title="Conversions"
        description="Every swap of mined value into a stablecoin, with its idempotency key."
        limit={limit}
        onLimitChange={setLimit}
        onRefresh={conversions.reload}
      >
        <div className="mb-3">
          <label className="form-check">
            <input
              className="form-check-input"
              type="checkbox"
              checked={onlyPending}
              onChange={(event) => setOnlyPending(event.target.checked)}
            />
            <span className="form-check-label ms-2">Only in-flight (pending or processing)</span>
          </label>
        </div>

        <DataTable
          columns={columns}
          rows={rows}
          lookups={lookups}
          loading={conversions.loading}
          error={conversions.error}
          onRetry={conversions.reload}
          emptyTitle="No conversions"
          emptyMessage="A settled pool payout is queued for conversion when a matching route exists."
          actions={(row) => (
            <button
              type="button"
              className="btn btn-outline-primary btn-xs"
              onClick={() => setSettling(row as unknown as ConversionTransaction)}
            >
              Settle
            </button>
          )}
        />
      </LedgerCard>

      {settling && (
        <SettleModal
          conversion={settling}
          onClose={() => setSettling(null)}
          onSaved={() => {
            setSettling(null);
            conversions.reload();
          }}
        />
      )}
    </>
  );
}

// --- Pool payouts ----------------------------------------------------------------------------

export function PoolPayoutsPage() {
  const [limit, setLimit] = useState(100);
  const payouts = useAsyncList<PoolPayout>(`/api/admin/pool-payouts?limit=${limit}`);
  const lookups = useLookups();

  const columns: Column[] = [
    {
      key: 'coinId',
      label: 'Coin',
      format: 'strong',
      render: (row, data: Lookups) => coinCode(data.coins, row.coinId),
    },
    { key: 'amount', label: 'Amount', align: 'end', render: (row) => formatNumber(row.amount as number) },
    { key: 'redeemedAmount', label: 'Attributed', align: 'end', render: (row) => formatNumber(row.redeemedAmount as number) },
    {
      key: 'remaining',
      label: 'Remaining',
      align: 'end',
      render: (row) => formatNumber((row.amount as number) - (row.redeemedAmount as number)),
    },
    {
      key: 'transactionHash',
      label: 'Transaction',
      format: 'mono',
      render: (row) => (row.transactionHash ? truncateMiddle(String(row.transactionHash), 12, 8) : '—'),
    },
    { key: 'confirmationsCount', label: 'Conf.', align: 'end', format: 'int' },
    { key: 'status', label: 'Status', format: 'badge' },
    { key: 'walletAddress', label: 'Wallet', format: 'mono', hideOnNarrow: true, render: (row) => truncateMiddle(row.walletAddress as string, 10, 6) },
    {
      key: 'receivedAt',
      label: 'Received',
      render: (row) => formatDateTime((row.receivedAt ?? row.createdAt) as string),
    },
  ];

  return (
    <LedgerCard
      title="Pool payouts"
      description="What each pool has actually paid out. Sheets are attributed to earned shares by reconciliation."
      limit={limit}
      onLimitChange={setLimit}
      onRefresh={payouts.reload}
    >
      <DataTable
        columns={columns}
        rows={payouts.data as unknown as Row[]}
        lookups={lookups}
        loading={payouts.loading}
        error={payouts.error}
        onRetry={payouts.reload}
        emptyTitle="No payouts recorded"
        emptyMessage="The pool monitor records a payout the first time it sees its transaction hash."
      />
    </LedgerCard>
  );
}

// --- Provider deposits -----------------------------------------------------------------------

export function ProviderDepositsPage() {
  const [limit, setLimit] = useState(100);
  const deposits = useAsyncList<ProviderDeposit>(`/api/admin/provider-deposits?limit=${limit}`);
  const lookups = useLookups();

  const columns: Column[] = [
    {
      key: 'llmProviderId',
      label: 'Provider',
      format: 'strong',
      render: (row, data: Lookups) =>
        data.llmProviders.find((provider) => provider.id === row.llmProviderId)?.name ?? '—',
    },
    { key: 'coinId', label: 'Coin', render: (row, data: Lookups) => coinCode(data.coins, row.coinId) },
    { key: 'network', label: 'Network' },
    { key: 'amount', label: 'Amount', align: 'end', render: (row) => formatUsd(row.amount as number) },
    {
      key: 'transactionHash',
      label: 'Transaction',
      format: 'mono',
      render: (row) => (row.transactionHash ? truncateMiddle(String(row.transactionHash), 12, 8) : '—'),
    },
    { key: 'confirmations', label: 'Conf.', align: 'end', format: 'int' },
    {
      key: 'providerCreditAfter',
      label: 'Credit after',
      align: 'end',
      render: (row) => formatUsd(row.providerCreditAfter as number | null),
    },
    { key: 'status', label: 'Status', format: 'badge' },
    { key: 'createdAt', label: 'Created', render: (row) => formatDateTime(String(row.createdAt)) },
    {
      key: 'error',
      label: 'Error',
      render: (row) =>
        row.error ? (
          <span className="fs-13" style={{ color: 'var(--tm-danger)' }}>
            {String(row.error)}
          </span>
        ) : (
          <span className="text-muted">—</span>
        ),
    },
  ];

  return (
    <LedgerCard
      title="Provider deposits"
      description="Stablecoin transfers into the LLM provider. A deposit is only credited once the provider's own balance grows."
      limit={limit}
      onLimitChange={setLimit}
      onRefresh={deposits.reload}
    >
      <DataTable
        columns={columns}
        rows={deposits.data as unknown as Row[]}
        lookups={lookups}
        loading={deposits.loading}
        error={deposits.error}
        onRetry={deposits.reload}
        emptyTitle="No deposits yet"
        emptyMessage="A completed conversion is transferred to the provider's deposit address automatically."
      />
    </LedgerCard>
  );
}
