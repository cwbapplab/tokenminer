import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api } from '../lib/api';
import type { Lookups } from '../lib/lookups';
import { formatDateTime, formatInt, formatRelative, humanize, truncateMiddle } from '../lib/format';
import type { Column, Row } from '../lib/resources';
import type { AdminHardwareSession, AdminUserDetail, AdminUserSummary } from '../lib/types';
import { Icon } from '../components/icons';
import { Badge, DataTable, ErrorState, Loading, StatCell, useAsyncList } from '../components/ui';

const LIMITS = [50, 100, 200, 500];

/** These screens render no reference data, so the table is given an empty lookup set. */
const NO_LOOKUPS: Lookups = {
  coins: [],
  conversionProviders: [],
  llmProviders: [],
  loading: false,
  error: null,
  reload: () => {},
};

interface AsyncObject<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  reload: () => void;
}

/** Loads a single object endpoint, mirroring useAsyncList for detail screens. */
function useAsyncObject<T>(path: string | null): AsyncObject<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(path !== null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  const reload = useCallback(() => setNonce((value) => value + 1), []);

  useEffect(() => {
    if (!path) {
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .get<T>(path)
      .then((result) => {
        if (!cancelled) {
          setData(result);
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : 'Could not load data.');
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [path, nonce]);

  return { data, loading, error, reload };
}

function sessionOf(row: Row): AdminHardwareSession | null {
  return (row.activeSession as AdminHardwareSession | null) ?? null;
}

// --- Users list ------------------------------------------------------------------------------

export function UsersPage() {
  const navigate = useNavigate();
  const [limit, setLimit] = useState(100);
  const [query, setQuery] = useState('');
  const users = useAsyncList<AdminUserSummary>(`/api/admin/users?limit=${limit}`);

  const rows = useMemo<Row[]>(() => {
    const needle = query.trim().toLowerCase();
    const all = users.data as unknown as Row[];

    return needle === '' ? all : all.filter((row) => JSON.stringify(row).toLowerCase().includes(needle));
  }, [users.data, query]);

  const columns: Column[] = [
    {
      key: 'id',
      label: 'Client ID',
      render: (row) => (
        <span className="cell-mono" title={String(row.id)}>
          {truncateMiddle(String(row.id), 10, 8)}
        </span>
      ),
    },
    {
      key: 'user',
      label: 'User',
      render: (row) => (
        <div>
          <div className="cell-strong">{row.displayName ? String(row.displayName) : String(row.email)}</div>
          {row.displayName ? <div className="field-hint mb-0">{String(row.email)}</div> : null}
        </div>
      ),
    },
    {
      key: 'hardwareCount',
      label: 'Hardware',
      align: 'end',
      render: (row) => (
        <Link
          to={`/users/${String(row.id)}/hardware`}
          className="btn btn-outline-primary btn-xs"
          title="View this user's hardware"
          onClick={(event) => event.stopPropagation()}
        >
          {formatInt(row.hardwareCount as number)}
        </Link>
      ),
    },
    {
      key: 'acceptedShares',
      label: 'Accepted',
      align: 'end',
      render: (row) => formatInt(row.acceptedShares as number),
    },
    {
      key: 'rejectedShares',
      label: 'Rejected',
      align: 'end',
      render: (row) => formatInt(row.rejectedShares as number),
    },
    { key: 'status', label: 'Status', format: 'badge' },
    { key: 'createdAt', label: 'Created', render: (row) => formatDateTime(String(row.createdAt)) },
  ];

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <h4 className="card-title">Users</h4>
          <div className="field-hint">
            Accounts with their device count and terminal share counts. Select a row to see the hardware
            in use; select a hardware count to open the device list.
          </div>
        </div>
        <div className="d-flex align-items-center gap-2">
          <div className="header-search">
            <Icon name="search" />
            <input
              className="form-control"
              placeholder="Filter users"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              aria-label="Filter users"
            />
          </div>
          <select
            className="form-select"
            style={{ width: '8rem' }}
            value={limit}
            onChange={(event) => setLimit(Number(event.target.value))}
            aria-label="Rows to load"
          >
            {LIMITS.map((value) => (
              <option key={value} value={value}>
                Last {value}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="btn btn-outline-secondary btn-square"
            onClick={users.reload}
            aria-label="Refresh"
            title="Refresh"
          >
            <Icon name="refresh" />
          </button>
        </div>
      </div>
      <div className="card-body pt-0">
        <DataTable
          columns={columns}
          rows={rows}
          lookups={NO_LOOKUPS}
          loading={users.loading}
          error={users.error}
          onRetry={users.reload}
          onRowClick={(row) => navigate(`/users/${String(row.id)}`)}
          emptyTitle="No users"
          emptyMessage="Accounts appear here once they register."
        />
      </div>
    </div>
  );
}

// --- User detail -----------------------------------------------------------------------------

export function UserDetailPage() {
  const { userId } = useParams<{ userId: string }>();
  const user = useAsyncObject<AdminUserDetail>(userId ? `/api/admin/users/${userId}` : null);

  if (user.loading) {
    return <Loading label="Loading the user" />;
  }

  if (user.error) {
    return <ErrorState message={user.error} onRetry={user.reload} />;
  }

  if (!user.data) {
    return null;
  }

  const detail = user.data;

  const columns: Column[] = [
    {
      key: 'hardware',
      label: 'Hardware',
      render: (row) => (
        <div>
          <div className="cell-strong">{row.name ? String(row.name) : 'Unnamed device'}</div>
          <div className="field-hint mb-0 cell-mono" title={String(row.hardwareId)}>
            {truncateMiddle(String(row.hardwareId), 10, 8)}
          </div>
        </div>
      ),
    },
    { key: 'status', label: 'Status', format: 'badge' },
    {
      key: 'mining',
      label: 'Mining',
      render: (row) => {
        const session = sessionOf(row);

        if (!session) {
          return <span className="text-muted">idle</span>;
        }

        return (
          <div>
            <div className="d-flex align-items-center gap-2">
              <Badge value={session.status} />
              <span className="cell-strong">{session.coinCode}</span>
            </div>
            <div className="field-hint mb-0">{session.poolName}</div>
          </div>
        );
      },
    },
    {
      key: 'startedAt',
      label: 'Since',
      render: (row) => {
        const session = sessionOf(row);

        return session ? (
          <span title={formatDateTime(session.startedAt)}>{formatRelative(session.startedAt)}</span>
        ) : (
          <span className="text-muted">—</span>
        );
      },
    },
    {
      key: 'acceptedShares',
      label: 'Accepted',
      align: 'end',
      render: (row) => formatInt(row.acceptedShares as number),
    },
    {
      key: 'rejectedShares',
      label: 'Rejected',
      align: 'end',
      render: (row) => formatInt(row.rejectedShares as number),
    },
  ];

  return (
    <>
      <div className="row tm-card-row">
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <div>
                <h4 className="card-title">{detail.displayName ?? detail.email}</h4>
                <div className="field-hint">
                  Client ID <span className="cell-mono">{detail.id}</span>
                </div>
              </div>
              <div className="d-flex align-items-center gap-2">
                <Link to="/users" className="btn btn-outline-secondary btn-sm">
                  All users
                </Link>
                <Link to={`/users/${detail.id}/hardware`} className="btn btn-outline-primary btn-sm">
                  Hardware list
                </Link>
              </div>
            </div>
            <div className="card-body pt-0">
              <div className="stat-strip">
                <StatCell label="Devices" value={formatInt(detail.hardwareCount)} icon="cpu" />
                <StatCell
                  label="Accepted shares"
                  value={formatInt(detail.acceptedShares)}
                  icon="check"
                  tone="success"
                />
                <StatCell
                  label="Rejected shares"
                  value={formatInt(detail.rejectedShares)}
                  icon="alert"
                  tone={detail.rejectedShares > 0 ? 'danger' : 'primary'}
                />
                <StatCell label="Status" value={humanize(detail.status)} icon="shield" tone="info" />
              </div>

              <div className="row mt-3">
                <div className="col-md-4">
                  <div className="field-hint mb-0">Email</div>
                  <div>{detail.email}</div>
                </div>
                <div className="col-md-4">
                  <div className="field-hint mb-0">Registered</div>
                  <div>{formatDateTime(detail.createdAt)}</div>
                </div>
                <div className="col-md-4">
                  <div className="field-hint mb-0">Last updated</div>
                  <div>{formatDateTime(detail.updatedAt)}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="row tm-card-row">
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <div>
                <h4 className="card-title">Hardware in use</h4>
                <div className="field-hint">Every device this account owns and what it is mining right now.</div>
              </div>
            </div>
            <div className="card-body pt-0">
              <DataTable
                columns={columns}
                rows={detail.hardware as unknown as Row[]}
                lookups={NO_LOOKUPS}
                loading={false}
                error={null}
                onRetry={user.reload}
                emptyTitle="No devices"
                emptyMessage="This account has not registered a mining device yet."
              />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// --- User hardware ---------------------------------------------------------------------------

export function UserHardwarePage() {
  const { userId } = useParams<{ userId: string }>();
  const user = useAsyncObject<AdminUserDetail>(userId ? `/api/admin/users/${userId}` : null);

  if (user.loading) {
    return <Loading label="Loading the hardware" />;
  }

  if (user.error) {
    return <ErrorState message={user.error} onRetry={user.reload} />;
  }

  if (!user.data) {
    return null;
  }

  const detail = user.data;

  const columns: Column[] = [
    {
      key: 'hardwareId',
      label: 'ID',
      render: (row) => (
        <span className="cell-mono" title={String(row.hardwareId)}>
          {truncateMiddle(String(row.hardwareId), 10, 8)}
        </span>
      ),
    },
    {
      key: 'name',
      label: 'Name',
      render: (row) => (row.name ? String(row.name) : <span className="text-muted">Unnamed</span>),
    },
    {
      key: 'connectedAt',
      label: 'Connected',
      render: (row) => (
        <span title={formatDateTime(row.connectedAt as string)}>
          {formatRelative(row.connectedAt as string)}
        </span>
      ),
    },
    { key: 'status', label: 'Status', format: 'badge' },
    {
      key: 'acceptedShares',
      label: 'Accepted',
      align: 'end',
      render: (row) => formatInt(row.acceptedShares as number),
    },
    {
      key: 'rejectedShares',
      label: 'Rejected',
      align: 'end',
      render: (row) => formatInt(row.rejectedShares as number),
    },
  ];

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <h4 className="card-title">Hardware</h4>
          <div className="field-hint">
            Devices owned by {detail.displayName ?? detail.email} ({formatInt(detail.hardwareCount)} total).
          </div>
        </div>
        <div className="d-flex align-items-center gap-2">
          <Link to={`/users/${detail.id}`} className="btn btn-outline-secondary btn-sm">
            User details
          </Link>
          <button
            type="button"
            className="btn btn-outline-secondary btn-square"
            onClick={user.reload}
            aria-label="Refresh"
            title="Refresh"
          >
            <Icon name="refresh" />
          </button>
        </div>
      </div>
      <div className="card-body pt-0">
        <DataTable
          columns={columns}
          rows={detail.hardware as unknown as Row[]}
          lookups={NO_LOOKUPS}
          loading={false}
          error={null}
          onRetry={user.reload}
          emptyTitle="No hardware"
          emptyMessage="This account has not registered a mining device yet."
        />
      </div>
    </div>
  );
}
