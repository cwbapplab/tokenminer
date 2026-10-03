import { useMemo, useState } from 'react';
import type { ChangeEvent } from 'react';
import type { Field, Row, SelectOption } from '../lib/resources';
import type { Lookups } from '../lib/lookups';
import { prettyJson } from '../lib/format';

export type FieldValues = Record<string, unknown>;

export type FormMode = 'create' | 'edit';

function resolveOptions(field: Field, lookups: Lookups): SelectOption[] {
  if (field.options) {
    return field.options;
  }

  switch (field.optionsFrom) {
    case 'coins':
      return lookups.coins.map((coin) => ({ value: coin.id, label: `${coin.code} · ${coin.name}` }));

    case 'conversionProviders':
      return lookups.conversionProviders.map((provider) => ({
        value: provider.id,
        label: `${provider.name} (priority ${provider.priority})`,
      }));

    case 'llmProviders':
      return lookups.llmProviders.map((provider) => ({ value: provider.id, label: provider.name }));

    default:
      return [];
  }
}

function readExisting(row: Row | null, field: Field): unknown {
  if (!row) {
    return field.defaultValue ?? (field.type === 'checkbox' ? false : field.type === 'multiselect' ? [] : '');
  }

  const container = field.nested ? (row[field.nested] as Row | null | undefined) : row;
  const value = container?.[field.name];

  if (value === null || value === undefined) {
    return field.type === 'checkbox' ? false : field.type === 'multiselect' ? [] : '';
  }

  if (field.type === 'json') {
    return prettyJson(value);
  }

  return value;
}

export function initialValues(fields: Field[], mode: FormMode, row: Row | null): FieldValues {
  const values: FieldValues = {};

  for (const field of fields) {
    if (mode === 'create' && field.editOnly) {
      continue;
    }

    if (mode === 'edit' && field.createOnly) {
      continue;
    }

    values[field.name] = readExisting(row, field);
  }

  return values;
}

/**
 * Turns form values into the payload the endpoint expects: `nested` fields are folded into
 * their container object, numbers are coerced, and JSON fields are parsed.
 */
export function buildPayload(
  fields: Field[],
  values: FieldValues,
  mode: FormMode,
  row: Row | null,
): Row {
  const payload: Row = {};

  for (const field of fields) {
    if (mode === 'create' && field.editOnly) {
      continue;
    }

    if (mode === 'edit' && field.createOnly) {
      continue;
    }

    let value = values[field.name];

    if (field.type === 'number' || field.type === 'decimal') {
      value = value === '' || value === null || value === undefined ? null : Number(value);
    } else if (field.type === 'json') {
      const text = typeof value === 'string' ? value.trim() : '';
      value = text.length === 0 ? null : JSON.parse(text);
    } else if (field.type === 'checkbox') {
      value = Boolean(value);
    } else if (field.type === 'multiselect') {
      value = Array.isArray(value) ? value : [];
    }

    if (field.nested) {
      // Fold every nested field into a single container. It is seeded from the existing row the
      // first time so an edit never blanks a sibling the form does not cover, then reused so the
      // next nested field adds to it rather than replacing it.
      const container =
        (payload[field.nested] as Row | undefined) ??
        { ...((row?.[field.nested] as Row | undefined) ?? {}) };
      container[field.name] = value;
      payload[field.nested] = container;
    } else {
      payload[field.name] = value;
    }
  }

  return payload;
}

export function validate(fields: Field[], values: FieldValues, mode: FormMode): Record<string, string> {
  const errors: Record<string, string> = {};

  for (const field of fields) {
    if (mode === 'create' && field.editOnly) {
      continue;
    }

    if (mode === 'edit' && field.createOnly) {
      continue;
    }

    const value = values[field.name];

    if (field.required) {
      const empty =
        value === null ||
        value === undefined ||
        value === '' ||
        (Array.isArray(value) && value.length === 0);

      if (empty) {
        errors[field.name] = `${field.label} is required.`;
        continue;
      }
    }

    if (field.type === 'json' && typeof value === 'string' && value.trim().length > 0) {
      try {
        JSON.parse(value);
      } catch {
        errors[field.name] = 'Not valid JSON.';
      }
    }

    if ((field.type === 'number' || field.type === 'decimal') && value !== '' && value !== null) {
      if (Number.isNaN(Number(value))) {
        errors[field.name] = 'Must be a number.';
      }
    }
  }

  return errors;
}

function FieldControl({
  field,
  value,
  error,
  lookups,
  onChange,
}: {
  field: Field;
  value: unknown;
  error?: string;
  lookups: Lookups;
  onChange: (value: unknown) => void;
}) {
  const options = useMemo(() => resolveOptions(field, lookups), [field, lookups]);
  const inputId = `field-${field.nested ?? ''}-${field.name}`;
  const invalid = error ? 'is-invalid' : '';

  if (field.type === 'checkbox') {
    return (
      <div className="form-check">
        <input
          className="form-check-input"
          type="checkbox"
          id={inputId}
          checked={Boolean(value)}
          onChange={(event: ChangeEvent<HTMLInputElement>) => onChange(event.target.checked)}
        />
        <label className="form-check-label ms-2" htmlFor={inputId}>
          {field.label}
        </label>
        {field.hint && <div className="field-hint">{field.hint}</div>}
      </div>
    );
  }

  return (
    <>
      <label className="form-label" htmlFor={inputId}>
        {field.label}
        {field.required && <span style={{ color: 'var(--tm-danger)' }}> *</span>}
      </label>

      {field.type === 'select' && (
        <select
          className={`form-select ${invalid}`}
          id={inputId}
          value={String(value ?? '')}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">Select…</option>
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      )}

      {field.type === 'multiselect' && (
        <div
          className={`form-control ${invalid}`}
          style={{ maxHeight: '11rem', overflowY: 'auto', padding: '0.625rem 0.875rem' }}
        >
          {options.length === 0 && <span className="text-muted">No options available.</span>}

          {options.map((option) => {
            const selected = Array.isArray(value) ? (value as string[]) : [];
            const checked = selected.includes(option.value);

            return (
              <div className="form-check" key={option.value}>
                <input
                  className="form-check-input"
                  type="checkbox"
                  id={`${inputId}-${option.value}`}
                  checked={checked}
                  onChange={(event) => {
                    const next = event.target.checked
                      ? [...selected, option.value]
                      : selected.filter((item) => item !== option.value);

                    onChange(next);
                  }}
                />
                <label className="form-check-label ms-2" htmlFor={`${inputId}-${option.value}`}>
                  {option.label}
                </label>
              </div>
            );
          })}
        </div>
      )}

      {field.type === 'textarea' && (
        <textarea
          className={`form-control ${invalid}`}
          id={inputId}
          value={String(value ?? '')}
          placeholder={field.placeholder}
          onChange={(event) => onChange(event.target.value)}
        />
      )}

      {field.type === 'json' && (
        <textarea
          className={`form-control code ${invalid}`}
          id={inputId}
          rows={8}
          spellCheck={false}
          value={String(value ?? '')}
          placeholder={field.placeholder}
          onChange={(event) => onChange(event.target.value)}
        />
      )}

      {(field.type === 'text' || field.type === 'number' || field.type === 'decimal') && (
        <input
          className={`form-control ${invalid}`}
          id={inputId}
          type={field.type === 'text' ? 'text' : 'number'}
          step={field.type === 'decimal' ? 'any' : undefined}
          value={String(value ?? '')}
          placeholder={field.placeholder}
          onChange={(event) => onChange(event.target.value)}
        />
      )}

      {field.hint && !error && <div className="field-hint">{field.hint}</div>}
      {error && <div className="field-error">{error}</div>}
    </>
  );
}

export function ResourceForm({
  fields,
  mode,
  values,
  errors,
  lookups,
  onChange,
}: {
  fields: Field[];
  mode: FormMode;
  values: FieldValues;
  errors: Record<string, string>;
  lookups: Lookups;
  onChange: (name: string, value: unknown) => void;
}) {
  const visible = fields.filter((field) => {
    if (mode === 'create' && field.editOnly) {
      return false;
    }

    return !(mode === 'edit' && field.createOnly);
  });

  return (
    <div className="row">
      {visible.map((field) => (
        <div className={field.fullWidth ? 'col-12' : 'col-md-6'} key={field.name} style={{ marginBottom: '1rem' }}>
          <FieldControl
            field={field}
            value={values[field.name]}
            error={errors[field.name]}
            lookups={lookups}
            onChange={(value) => onChange(field.name, value)}
          />
        </div>
      ))}
    </div>
  );
}

/** Small helper so screens can report the first error without repeating the fallback. */
export function firstError(errors: Record<string, string>): string | null {
  const entry = Object.values(errors).find(Boolean);

  return entry ?? null;
}

export function useFormState(fields: Field[], mode: FormMode, row: Row | null) {
  const [values, setValues] = useState<FieldValues>(() => initialValues(fields, mode, row));
  const [errors, setErrors] = useState<Record<string, string>>({});

  function update(name: string, value: unknown) {
    setValues((current) => ({ ...current, [name]: value }));
    setErrors((current) => {
      if (!current[name]) {
        return current;
      }

      const next = { ...current };
      delete next[name];

      return next;
    });
  }

  return { values, errors, setErrors, update };
}
