import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { Icon } from './icons';
import type { IconName } from './icons';
import { statusTone } from '../lib/format';
import { api } from '../lib/api';
import type { Lookups } from '../lib/lookups';
import type { Column, Row } from '../lib/resources';

// --- Badges ---------------------------------------------------------------------------------

export function Badge({ value, tone }: { value: string; tone?: string }) {
  return <span className={`badge badge-${tone ?? statusTone(value)}`}>{value.replace(/_/g, ' ')}</span>;
}

// --- Feedback -------------------------------------------------------------------------------

export function Loading({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="tm-placeholder">
      <div className="spinner lg" />
      <p>{label}…</p>
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="tm-placeholder">
      <div className="placeholder-icon" style={{ background: 'rgba(243,64,64,0.1)', color: 'var(--tm-danger)' }}>
        <Icon name="alert" />
      </div>
      <h5>Something went wrong</h5>
      <p>{message}</p>
      {onRetry && (
        <button type="button" className="btn btn-outline-primary btn-sm" onClick={onRetry}>
          <Icon name="refresh" />
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyState({
  icon = 'inbox',
  title,
  message,
  action,
}: {
  icon?: IconName;
  title: string;
  message: string;
  action?: ReactNode;
}) {
  return (
    <div className="tm-placeholder">
      <div className="placeholder-icon">
        <Icon name={icon} />
      </div>
      <h5>{title}</h5>
      <p>{message}</p>
      {action}
    </div>
  );
}

// --- Toasts ---------------------------------------------------------------------------------

type ToastTone = 'success' | 'error' | 'warning' | 'info';

interface Toast {
  id: number;
  message: string;
  tone: ToastTone;
}

interface ToastContextValue {
  push: (message: string, tone?: ToastTone) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

let toastSequence = 0;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const push = useCallback((message: string, tone: ToastTone = 'success') => {
    toastSequence += 1;
    const id = toastSequence;

    setToasts((current) => [...current, { id, message, tone }]);

    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
    }, 5000);
  }, []);

  const value = useMemo<ToastContextValue>(() => ({ push }), [push]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="tm-toasts" role="status" aria-live="polite">
        {toasts.map((toast) => (
          <div key={toast.id} className={`tm-toast ${toast.tone}`}>
            <Icon name={toast.tone === 'error' ? 'alert' : 'check'} />
            <div className="toast-body">{toast.message}</div>
            <button
              type="button"
              className="modal-close"
              aria-label="Dismiss"
              onClick={() => setToasts((current) => current.filter((item) => item.id !== toast.id))}
            >
              <Icon name="close" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const context = useContext(ToastContext);

  if (!context) {
    throw new Error('useToast must be used inside a ToastProvider.');
  }

  return context;
}

// --- Modal ----------------------------------------------------------------------------------

export function Modal({
  title,
  subtitle,
  onClose,
  children,
  footer,
  size = 'default',
}: {
  title: string;
  subtitle?: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  size?: 'narrow' | 'default' | 'wide';
}) {
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        onClose();
      }
    }

    document.addEventListener('keydown', onKeyDown);
    document.body.style.overflow = 'hidden';

    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.body.style.overflow = '';
    };
  }, [onClose]);

  const sizeClass = size === 'default' ? '' : size;

  return (
    <div
      className="tm-modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label={title}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div className={`tm-modal ${sizeClass}`}>
        <div className="modal-header">
          <div>
            <h4 className="modal-title">{title}</h4>
            {subtitle && <div className="field-hint">{subtitle}</div>}
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            <Icon name="close" />
          </button>
        </div>
        <div className="modal-body">{children}</div>
        {footer && <div className="modal-footer">{footer}</div>}
      </div>
    </div>
  );
}

export function ConfirmDialog({
  title,
  message,
  confirmLabel = 'Confirm',
  tone = 'primary',
  busy = false,
  onConfirm,
  onCancel,
}: {
  title: string;
  message: string;
  confirmLabel?: string;
  tone?: 'primary' | 'danger';
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <Modal
      title={title}
      size="narrow"
      onClose={onCancel}
      footer={
        <>
          <button type="button" className="btn btn-outline-secondary" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
          <button
            type="button"
            className={tone === 'danger' ? 'btn btn-outline-danger' : 'btn btn-primary'}
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? 'Working…' : confirmLabel}
          </button>
        </>
      }
    >
      <p style={{ margin: 0 }}>{message}</p>
    </Modal>
  );
}

// --- Stat strip -----------------------------------------------------------------------------

/**
 * One stat inside a `.stat-strip`. It deliberately brings no card of its own: the strip is a
 * single card and the dividers do the separating.
 */
export function StatCell({
  label,
  value,
  icon,
  tone = 'primary',
  hint,
}: {
  label: string;
  value: string;
  icon: IconName;
  tone?: 'primary' | 'success' | 'info' | 'warning' | 'danger';
  hint?: string;
}) {
  return (
    <div className="stat-cell">
      <div>
        <h2 className="stat-value">{value}</h2>
        <span className="stat-label">{label}</span>
        {hint && <span className="field-hint">{hint}</span>}
      </div>
      <div className={`stat-icon ${tone === 'primary' ? '' : tone}`}>
        <Icon name={icon} />
      </div>
    </div>
  );
}

// --- Table ----------------------------------------------------------------------------------

function readPath(row: Row, path: string): unknown {
  return path.split('.').reduce<unknown>((current, segment) => {
    if (current && typeof current === 'object') {
      return (current as Row)[segment];
    }

    return undefined;
  }, row);
}

export function renderCell(column: Column, row: Row, lookups: Lookups): ReactNode {
  if (column.render) {
    return column.render(row, lookups);
  }

  const value = readPath(row, column.key);

  if (value === null || value === undefined || value === '') {
    return <span className="text-muted">—</span>;
  }

  switch (column.format) {
    case 'badge':
      return <Badge value={String(value)} />;

    case 'strong':
      return <span className="cell-strong">{String(value)}</span>;

    case 'mono':
      return <span className="cell-mono">{String(value)}</span>;

    case 'bool':
      return value ? 'Yes' : 'No';

    case 'date':
      return new Date(String(value)).toLocaleDateString();

    case 'datetime':
      return new Date(String(value)).toLocaleString();

    case 'relative':
      return formatRelativeValue(String(value));

    case 'usd':
      return formatUsdValue(value as number);

    case 'number':
      return formatNumberValue(value as number);

    case 'int':
      return Number(value).toLocaleString();

    case 'coinCodes':
      return String(value);

    default:
      return String(value);
  }
}

function formatRelativeValue(value: string): string {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return '—';
  }

  const seconds = Math.round((Date.now() - date.getTime()) / 1000);

  if (seconds < 60) {
    return `${Math.max(seconds, 0)}s ago`;
  }

  const minutes = Math.round(seconds / 60);

  if (minutes < 60) {
    return `${minutes}m ago`;
  }

  const hours = Math.round(minutes / 60);

  if (hours < 24) {
    return `${hours}h ago`;
  }

  return `${Math.round(hours / 24)}d ago`;
}

function formatUsdValue(value: number): string {
  if (value === null || value === undefined) {
    return '—';
  }

  return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 })}`;
}

function formatNumberValue(value: number): string {
  if (value === null || value === undefined) {
    return '—';
  }

  return value.toLocaleString(undefined, { maximumFractionDigits: 8 });
}

export function DataTable({
  columns,
  rows,
  lookups,
  loading,
  error,
  onRetry,
  rowKey = 'id',
  actions,
  emptyTitle = 'Nothing here yet',
  emptyMessage = 'Once data exists it will appear here.',
  emptyAction,
}: {
  columns: Column[];
  rows: Row[];
  lookups: Lookups;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
  rowKey?: string;
  actions?: (row: Row) => ReactNode;
  emptyTitle?: string;
  emptyMessage?: string;
  emptyAction?: ReactNode;
}) {
  if (loading) {
    return <Loading />;
  }

  if (error) {
    return <ErrorState message={error} onRetry={onRetry} />;
  }

  if (rows.length === 0) {
    return <EmptyState title={emptyTitle} message={emptyMessage} action={emptyAction} />;
  }

  return (
    <div className="table-responsive">
      <table className="table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th
                key={column.key}
                style={column.align === 'end' ? { textAlign: 'right' } : undefined}
              >
                {column.label}
              </th>
            ))}
            {actions && <th style={{ textAlign: 'right' }}>Actions</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={String(row[rowKey] ?? index)}>
              {columns.map((column) => (
                <td
                  key={column.key}
                  style={column.align === 'end' ? { textAlign: 'right' } : undefined}
                >
                  {renderCell(column, row, lookups)}
                </td>
              ))}
              {actions && <td style={{ textAlign: 'right' }}>{actions(row)}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// --- Data loading hook ----------------------------------------------------------------------

export interface AsyncList<T> {
  data: T[];
  loading: boolean;
  error: string | null;
  reload: () => void;
}

/** Loads a list endpoint and exposes a reload, which every screen action needs. */
export function useAsyncList<T>(path: string | null): AsyncList<T> {
  const [data, setData] = useState<T[]>([]);
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
      .get<T[]>(path)
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
