/**
 * Presentation helpers. Values come from the API in invariant form; everything an operator
 * reads is formatted here so the screens stay free of formatting noise.
 */

const dateTimeFormatter = new Intl.DateTimeFormat(undefined, {
  year: 'numeric',
  month: 'short',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
});

const dateFormatter = new Intl.DateTimeFormat(undefined, {
  year: 'numeric',
  month: 'short',
  day: '2-digit',
});

export function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return '—';
  }

  const date = new Date(value);

  return Number.isNaN(date.getTime()) ? '—' : dateTimeFormatter.format(date);
}

export function formatDate(value: string | null | undefined): string {
  if (!value) {
    return '—';
  }

  const date = new Date(value);

  return Number.isNaN(date.getTime()) ? '—' : dateFormatter.format(date);
}

/** Relative age, which is what matters for heartbeats and in-flight work. */
export function formatRelative(value: string | null | undefined): string {
  if (!value) {
    return 'never';
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return 'never';
  }

  const seconds = Math.round((Date.now() - date.getTime()) / 1000);

  if (seconds < 0) {
    return 'just now';
  }

  if (seconds < 60) {
    return `${seconds}s ago`;
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

export function formatUsd(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined) {
    return '—';
  }

  return `$${value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: Math.max(digits, 6),
  })}`;
}

export function formatNumber(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined) {
    return '—';
  }

  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function formatInt(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return '—';
  }

  return value.toLocaleString();
}

/** `native_auto_payout` reads better as `Native auto payout`. */
export function humanize(value: string | null | undefined): string {
  if (!value) {
    return '—';
  }

  const spaced = value.replace(/[_-]+/g, ' ').trim();

  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

export function truncateMiddle(value: string | null | undefined, head = 10, tail = 8): string {
  if (!value) {
    return '—';
  }

  if (value.length <= head + tail + 1) {
    return value;
  }

  return `${value.slice(0, head)}…${value.slice(-tail)}`;
}

export function initials(value: string | null | undefined): string {
  if (!value) {
    return '?';
  }

  const source = value.includes('@') ? value.split('@')[0] : value;
  const parts = source.split(/[.\s_-]+/).filter(Boolean);

  if (parts.length === 0) {
    return '?';
  }

  return (parts[0].charAt(0) + (parts[1]?.charAt(0) ?? '')).toUpperCase();
}

/**
 * Maps a status onto a badge tone. Kept in one place so the same state never shows up in two
 * different colours across screens.
 */
export function statusTone(status: string | null | undefined): string {
  switch ((status ?? '').toLowerCase()) {
    case 'active':
    case 'confirmed':
    case 'credited':
    case 'completed':
    case 'accepted':
    case 'redeemed':
    case 'online':
      return 'success';

    case 'pending':
    case 'processing':
    case 'broadcast':
    case 'awaiting_confirmations':
    case 'paused':
      return 'warning';

    case 'failed':
    case 'rejected':
    case 'disabled':
    case 'cancelled':
      return 'danger';

    default:
      return 'secondary';
  }
}

export function prettyJson(value: unknown): string {
  if (value === null || value === undefined) {
    return '';
  }

  if (typeof value === 'string') {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value;
    }
  }

  return JSON.stringify(value, null, 2);
}
