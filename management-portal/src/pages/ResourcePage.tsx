import { useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import { api } from '../lib/api';
import { useLookups } from '../lib/lookups';
import type { Lookups } from '../lib/lookups';
import { findResource } from '../lib/resources';
import type { ActionKind, Field, Resource, Row } from '../lib/resources';
import { formatDateTime, formatUsd, humanize } from '../lib/format';
import { Icon } from '../components/icons';
import { DataTable, EmptyState, Modal, useAsyncList, useToast } from '../components/ui';
import { ResourceForm, buildPayload, useFormState, validate } from '../components/ResourceForm';
import type { FormMode } from '../components/ResourceForm';
import type { LlmProviderDepositAccount, LlmProviderModel } from '../lib/types';

// --- Create / edit ---------------------------------------------------------------------------

function ResourceFormModal({
  resource,
  mode,
  row,
  lookups,
  onClose,
  onSaved,
}: {
  resource: Resource;
  mode: FormMode;
  row: Row | null;
  lookups: Lookups;
  onClose: () => void;
  onSaved: () => void;
}) {
  const { values, errors, setErrors, update } = useFormState(resource.fields, mode, row);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const { push } = useToast();

  async function submit() {
    const validation = validate(resource.fields, values, mode);

    if (Object.keys(validation).length > 0) {
      setErrors(validation);
      return;
    }

    setBusy(true);
    setFailure(null);

    try {
      const payload = buildPayload(resource.fields, values, mode, row);

      if (mode === 'create') {
        await api.post(resource.path, payload);
      } else if (resource.updatePath && row) {
        await api.put(resource.updatePath(row), payload);
      } else if (row) {
        await api.put(`${resource.path}/${String(row.id)}`, payload);
      }

      push(`${humanize(resource.singular)} ${mode === 'create' ? 'created' : 'updated'}.`);
      onSaved();
    } catch (caught) {
      setFailure(caught instanceof Error ? caught.message : 'Could not save.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal
      title={mode === 'create' ? `New ${resource.singular}` : `Edit ${resource.singular}`}
      subtitle={resource.description}
      onClose={onClose}
      footer={
        <>
          <button type="button" className="btn btn-outline-secondary" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button type="button" className="btn btn-primary" onClick={submit} disabled={busy}>
            {busy ? 'Saving…' : mode === 'create' ? 'Create' : 'Save changes'}
          </button>
        </>
      }
    >
      {failure && <div className="alert alert-danger mb-3">{failure}</div>}

      <ResourceForm
        fields={resource.fields}
        mode={mode}
        values={values}
        errors={errors}
        lookups={lookups}
        onChange={update}
      />
    </Modal>
  );
}

// --- Provider deposit rails ------------------------------------------------------------------

function DepositAccountsModal({
  providerId,
  accounts,
  lookups,
  onClose,
  onSaved,
}: {
  providerId: string;
  accounts: LlmProviderDepositAccount[];
  lookups: Lookups;
  onClose: () => void;
  onSaved: () => void;
}) {
  const fields: Field[] = [
    { name: 'coinId', label: 'Coin', type: 'select', required: true, optionsFrom: 'coins' },
    { name: 'network', label: 'Network', type: 'text', required: true, placeholder: 'bep20' },
    { name: 'depositAddress', label: 'Deposit address', type: 'text', required: true },
    { name: 'status', label: 'Status', type: 'select', required: true, options: [
      { value: 'active', label: 'Active' },
      { value: 'disabled', label: 'Disabled' },
    ], defaultValue: 'active' },
  ];

  const { values, errors, setErrors, update } = useFormState(fields, 'create', null);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const { push } = useToast();

  async function save() {
    const validation = validate(fields, values, 'create');

    if (Object.keys(validation).length > 0) {
      setErrors(validation);
      return;
    }

    setBusy(true);
    setFailure(null);

    try {
      await api.put(`/api/admin/llm-providers/${providerId}/deposit-accounts`, buildPayload(fields, values, 'create', null));
      push('Deposit rail saved.');
      onSaved();
    } catch (caught) {
      setFailure(caught instanceof Error ? caught.message : 'Could not save the deposit rail.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal
      title="Deposit rails"
      subtitle="The address funds are sent to for each coin and network. The pair identifies the rail."
      onClose={onClose}
      footer={
        <>
          <button type="button" className="btn btn-outline-secondary" onClick={onClose} disabled={busy}>
            Close
          </button>
          <button type="button" className="btn btn-primary" onClick={save} disabled={busy}>
            {busy ? 'Saving…' : 'Save rail'}
          </button>
        </>
      }
    >
      {failure && <div className="alert alert-danger mb-3">{failure}</div>}

      {accounts.length > 0 && (
        <div className="table-responsive mb-4">
          <table className="table">
            <thead>
              <tr>
                <th>Coin</th>
                <th>Network</th>
                <th>Address</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {accounts.map((account) => (
                <tr key={account.id}>
                  <td className="cell-strong">{account.coinCode}</td>
                  <td>{account.network}</td>
                  <td className="cell-mono">{account.depositAddress}</td>
                  <td>{humanize(account.status)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {accounts.length === 0 && (
        <div className="alert alert-warning mb-4">
          This provider has no deposit rails yet, so the deposit pipeline has nowhere to send funds.
        </div>
      )}

      <ResourceForm fields={fields} mode="create" values={values} errors={errors} lookups={lookups} onChange={update} />
    </Modal>
  );
}

// --- Provider model catalogue ----------------------------------------------------------------

function ModelsModal({ providerId, onClose }: { providerId: string; onClose: () => void }) {
  const models = useAsyncList<LlmProviderModel>(`/api/admin/llm-providers/${providerId}/models`);
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();

    if (needle.length === 0) {
      return models.data;
    }

    return models.data.filter((model) =>
      `${model.modelId} ${model.name ?? ''}`.toLowerCase().includes(needle),
    );
  }, [models.data, query]);

  return (
    <Modal
      title="Model catalogue"
      subtitle="Mirrored from the provider. Models it no longer lists are disabled here rather than deleted."
      size="wide"
      onClose={onClose}
      footer={
        <button type="button" className="btn btn-outline-secondary" onClick={onClose}>
          Close
        </button>
      }
    >
      <div className="header-search mb-3" style={{ maxWidth: '100%' }}>
        <Icon name="search" />
        <input
          type="search"
          placeholder="Search models"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>

      {models.loading && <div className="spinner lg mx-auto" />}

      {models.error && <div className="alert alert-danger">{models.error}</div>}

      {!models.loading && !models.error && filtered.length === 0 && (
        <EmptyState
          icon="package"
          title="No models"
          message="The catalogue syncs from the provider on a schedule. Nothing has been synced yet."
        />
      )}

      {filtered.length > 0 && (
        <div className="table-responsive" style={{ maxHeight: '28rem', overflowY: 'auto' }}>
          <table className="table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Context</th>
                <th style={{ textAlign: 'right' }}>Input / 1M</th>
                <th style={{ textAlign: 'right' }}>Output / 1M</th>
                <th>Status</th>
                <th>Synced</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((model) => (
                <tr key={model.id}>
                  <td>
                    <span className="cell-strong">{model.modelId}</span>
                    {model.name && <div className="field-hint">{model.name}</div>}
                  </td>
                  <td>{model.contextLength?.toLocaleString() ?? '—'}</td>
                  <td style={{ textAlign: 'right' }}>{formatUsd(model.inputCost)}</td>
                  <td style={{ textAlign: 'right' }}>{formatUsd(model.outputCost)}</td>
                  <td>{humanize(model.status)}</td>
                  <td className="fs-13 text-muted">{formatDateTime(model.lastSyncedAt)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Modal>
  );
}

// --- Screen ----------------------------------------------------------------------------------

export function ResourcePage() {
  const { resourceKey } = useParams<{ resourceKey: string }>();
  const resource = findResource(resourceKey);
  const lookups = useLookups();
  const list = useAsyncList<Row>(resource?.path ?? null);
  const [query, setQuery] = useState('');
  const [formModal, setFormModal] = useState<{ mode: FormMode; row: Row | null } | null>(null);
  const [actionModal, setActionModal] = useState<{ kind: ActionKind; row: Row } | null>(null);

  const rows = list.data;

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();

    if (needle.length === 0) {
      return rows;
    }

    return rows.filter((row) => JSON.stringify(row).toLowerCase().includes(needle));
  }, [rows, query]);

  function reloadAll() {
    list.reload();
    lookups.reload();
  }

  if (!resource) {
    return (
      <EmptyState
        icon="alert"
        title="Unknown screen"
        message="That resource is not registered in the portal."
      />
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <h4 className="card-title">{resource.title}</h4>
          <div className="field-hint">{resource.description}</div>
        </div>
        <div className="d-flex align-items-center gap-2">
          <div className="header-search" style={{ minWidth: '14rem' }}>
            <Icon name="search" />
            <input
              type="search"
              placeholder={`Search ${resource.title.toLowerCase()}`}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>
          <button
            type="button"
            className="btn btn-outline-secondary btn-square"
            onClick={reloadAll}
            aria-label="Refresh"
            title="Refresh"
          >
            <Icon name="refresh" />
          </button>
          {resource.canCreate && (
            <button type="button" className="btn btn-primary" onClick={() => setFormModal({ mode: 'create', row: null })}>
              <Icon name="plus" />
              New
            </button>
          )}
        </div>
      </div>

      <div className="card-body pt-0">
        <DataTable
          columns={resource.columns}
          rows={filtered}
          lookups={lookups}
          loading={list.loading || lookups.loading}
          error={list.error}
          onRetry={reloadAll}
          emptyTitle={query ? 'No matches' : `No ${resource.title.toLowerCase()} yet`}
          emptyMessage={
            query
              ? 'Try a different search term.'
              : `${resource.singular.charAt(0).toUpperCase()}${resource.singular.slice(1)} created here will show up for the mining and treasury pipelines.`
          }
          emptyAction={
            resource.canCreate && !query ? (
              <button type="button" className="btn btn-primary btn-sm" onClick={() => setFormModal({ mode: 'create', row: null })}>
                <Icon name="plus" />
                Create the first one
              </button>
            ) : undefined
          }
          actions={(row) => (
            <div className="table-actions">
              {(resource.actions ?? []).map((action) => (
                <button
                  key={action.kind}
                  type="button"
                  className="btn btn-outline-secondary btn-xs"
                  onClick={() => setActionModal({ kind: action.kind, row })}
                >
                  {action.label}
                </button>
              ))}
              {resource.canEdit && (
                <button
                  type="button"
                  className="btn btn-outline-primary btn-icon"
                  aria-label="Edit"
                  title="Edit"
                  onClick={() => setFormModal({ mode: 'edit', row })}
                >
                  <Icon name="edit" />
                </button>
              )}
            </div>
          )}
        />
      </div>

      {formModal && (
        <ResourceFormModal
          key={`${formModal.mode}-${String(formModal.row?.id ?? 'new')}`}
          resource={resource}
          mode={formModal.mode}
          row={formModal.row}
          lookups={lookups}
          onClose={() => setFormModal(null)}
          onSaved={() => {
            setFormModal(null);
            reloadAll();
          }}
        />
      )}

      {actionModal?.kind === 'depositAccounts' && (
        <DepositAccountsModal
          providerId={String(actionModal.row.id)}
          accounts={(actionModal.row.depositAccounts as LlmProviderDepositAccount[]) ?? []}
          lookups={lookups}
          onClose={() => setActionModal(null)}
          onSaved={() => {
            setActionModal(null);
            reloadAll();
          }}
        />
      )}

      {actionModal?.kind === 'models' && (
        <ModelsModal providerId={String(actionModal.row.id)} onClose={() => setActionModal(null)} />
      )}
    </div>
  );
}
