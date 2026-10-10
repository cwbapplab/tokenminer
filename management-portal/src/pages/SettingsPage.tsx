import { useMemo, useState } from 'react';
import { api } from '../lib/api';
import { formatDateTime } from '../lib/format';
import type { Column, Row } from '../lib/resources';
import { Icon } from '../components/icons';
import { DataTable, Modal, useAsyncList, useToast } from '../components/ui';
import type { SystemConfiguration } from '../lib/types';

/**
 * Settings the background services read at runtime. Values are stored as JSON, so a bare
 * `5.0` or `true` is what goes in the box.
 */
const KNOWN_KEYS: Record<string, { label: string; hint: string }> = {
  active_llm_provider_id: {
    label: 'Active LLM provider',
    hint: 'Guid of the provider the pipeline tops up.',
  },
  active_conversion_provider_id: {
    label: 'Active conversion provider',
    hint: 'Guid of the provider used for swaps.',
  },
  active_pool_id: {
    label: 'Active pool',
    hint: 'Guid of the pool currently being mined.',
  },
  pool_monitor_interval_seconds: {
    label: 'Pool monitor interval',
    hint: 'Seconds between pool balance and payout-history polls.',
  },
  coin_price_refresh_interval_seconds: {
    label: 'Coin price refresh interval',
    hint: 'Seconds between price refreshes.',
  },
  payout_confirmation_threshold: {
    label: 'Payout confirmation threshold',
    hint: 'Confirmations before a payout counts as settled.',
  },
  provider_balance_target: {
    label: 'Provider balance target',
    hint: 'USD balance the auto top-up aims for.',
  },
  provider_balance_minimum: {
    label: 'Provider balance minimum',
    hint: 'USD balance that triggers a top-up.',
  },
  minimum_conversion_amount: {
    label: 'Minimum conversion amount',
    hint: 'Smallest conversion worth performing.',
  },
  minimum_payout_amount: {
    label: 'Minimum payout amount',
    hint: 'Smallest payout worth requesting from a pool.',
  },
  auto_top_up_enabled: {
    label: 'Auto top-up enabled',
    hint: 'true or false. Turns the top-up state machine on and off.',
  },
};

function ConfigurationModal({
  existing,
  onClose,
  onSaved,
}: {
  existing: SystemConfiguration | null;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [key, setKey] = useState(existing?.key ?? '');
  const [value, setValue] = useState(existing?.value ?? '');
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const { push } = useToast();

  const meta = KNOWN_KEYS[key];

  async function submit() {
    if (key.trim().length === 0) {
      setFailure('A key is required.');
      return;
    }

    setBusy(true);
    setFailure(null);

    try {
      await api.put(`/api/admin/system/configurations/${encodeURIComponent(key.trim())}`, {
        value: value.trim(),
      });

      push('Configuration saved.');
      onSaved();
    } catch (caught) {
      setFailure(caught instanceof Error ? caught.message : 'Could not save the setting.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal
      title={existing ? 'Edit setting' : 'Set a setting'}
      subtitle="Values are JSON literals, for example 5.0, true or &quot;active&quot;."
      onClose={onClose}
      footer={
        <>
          <button type="button" className="btn btn-outline-secondary" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="button" className="btn btn-primary" onClick={submit} disabled={busy}>
            {busy ? 'Saving…' : 'Save'}
          </button>
        </>
      }
    >
      {failure && <div className="alert alert-danger mb-3">{failure}</div>}

      <div className="mb-3">
        <label className="form-label" htmlFor="config-key">
          Key
        </label>
        {existing ? (
          <input className="form-control" id="config-key" value={key} readOnly />
        ) : (
          <>
            <select
              className="form-select mb-2"
              value={KNOWN_KEYS[key] ? key : ''}
              onChange={(event) => setKey(event.target.value)}
            >
              <option value="">Choose a known setting…</option>
              {Object.entries(KNOWN_KEYS).map(([knownKey, description]) => (
                <option key={knownKey} value={knownKey}>
                  {description.label} ({knownKey})
                </option>
              ))}
            </select>
            <input
              className="form-control"
              aria-label="Custom key"
              placeholder="…or type a custom key"
              value={key}
              onChange={(event) => setKey(event.target.value)}
            />
          </>
        )}
        {meta ? <div className="field-hint">{meta.hint}</div> : null}
      </div>

      <div className="mb-3">
        <label className="form-label" htmlFor="config-value">
          Value
        </label>
        <textarea
          className="form-control code"
          id="config-value"
          rows={4}
          spellCheck={false}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="5.0"
        />
        <div className="field-hint">Leave empty to clear it and fall back to the configured default.</div>
      </div>
    </Modal>
  );
}

export function SettingsPage() {
  const configurations = useAsyncList<SystemConfiguration>('/api/admin/system/configurations');
  const [editing, setEditing] = useState<{ config: SystemConfiguration | null } | null>(null);

  const columns: Column[] = useMemo(
    () => [
      {
        key: 'key',
        label: 'Setting',
        format: 'strong',
        render: (row) => {
          const key = String(row.key);
          const meta = KNOWN_KEYS[key];

          return (
            <div>
              <span className="cell-strong">{meta?.label ?? key}</span>
              <div className="field-hint cell-mono">{key}</div>
            </div>
          );
        },
      },
      { key: 'value', label: 'Value', format: 'mono' },
      { key: 'updatedAt', label: 'Updated', render: (row) => formatDateTime(String(row.updatedAt)) },
    ],
    [],
  );

  return (
    <>
      <div className="card">
        <div className="card-header">
          <div>
            <h4 className="card-title">System configuration</h4>
            <div className="field-hint">
              Runtime settings that override the values in configuration files. Unset keys fall back to the
              deployed defaults.
            </div>
          </div>
          <div className="d-flex align-items-center gap-2">
            <button type="button" className="btn btn-outline-secondary btn-square" onClick={configurations.reload} aria-label="Refresh" title="Refresh">
              <Icon name="refresh" />
            </button>
            <button type="button" className="btn btn-primary" onClick={() => setEditing({ config: null })}>
              <Icon name="plus" />
              Set a value
            </button>
          </div>
        </div>

        <div className="card-body pt-0">
          <DataTable
            columns={columns}
            rows={configurations.data as unknown as Row[]}
            lookups={{ coins: [], conversionProviders: [], llmProviders: [], loading: false, error: null, reload: () => {} }}
            loading={configurations.loading}
            error={configurations.error}
            onRetry={configurations.reload}
            rowKey="key"
            emptyTitle="Nothing overridden"
            emptyMessage="Every service is running on its deployed defaults. Set a value to override one."
            emptyAction={
              <button type="button" className="btn btn-primary btn-sm" onClick={() => setEditing({ config: null })}>
                <Icon name="plus" />
                Set a value
              </button>
            }
            actions={(row) => (
              <div className="table-actions">
                <button
                  type="button"
                  className="btn btn-outline-primary btn-icon"
                  aria-label="Edit"
                  title="Edit"
                  onClick={() => setEditing({ config: row as unknown as SystemConfiguration })}
                >
                  <Icon name="edit" />
                </button>
              </div>
            )}
          />
        </div>
      </div>

      {editing && (
        <ConfigurationModal
          key={editing.config?.key ?? 'new'}
          existing={editing.config}
          onClose={() => setEditing(null)}
          onSaved={() => {
            setEditing(null);
            configurations.reload();
          }}
        />
      )}
    </>
  );
}
