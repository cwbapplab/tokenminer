export type ClassValue = string | false | null | undefined;

/** Minimal class-name joiner (no external dependency). */
export function cn(...values: ClassValue[]): string {
  return values.filter(Boolean).join(" ");
}

export function formatUsd(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "—";
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);
}

export function formatHashrate(hashesPerSecond: number | null | undefined): string {
  if (hashesPerSecond === null || hashesPerSecond === undefined || hashesPerSecond <= 0) {
    return "0 H/s";
  }
  const units: Array<[number, string]> = [
    [1e12, "TH/s"],
    [1e9, "GH/s"],
    [1e6, "MH/s"],
    [1e3, "KH/s"],
  ];
  for (const [scale, unit] of units) {
    if (hashesPerSecond >= scale) {
      return `${(hashesPerSecond / scale).toFixed(2)} ${unit}`;
    }
  }
  return `${hashesPerSecond.toFixed(2)} H/s`;
}

/**
 * Pearl's hashrate, in TH/s — the unit every other Pearl miner reports.
 *
 * The backend does the conversion, not this: one candidate tile is one 16×16 output of the full
 * k-deep GEMM, so it is `16 * 16 * k` MACs and `tiles/s * 16 * 16 * k / 1e12` is TH/s. That is the
 * reference miner's own arithmetic — `TH_PER_MTILE = (1 << 20) / 1e6`, annotated "1 Mtile/s ~= 1.0486
 * TH/s" (`tests/bench_split.py:20`) — and its `1 << 20` is just `16 * 16 * 4096` written out. This
 * function only chooses a suffix and a number of decimals.
 *
 * It must not be folded into `formatHashrate`. That one is a Quantus H/s counter whose "TH/s" is
 * 10^12 of *Quantus* hashes; a Pearl "TH" is 10^12 of int8 MACs, which is not a Quantus hash, so the
 * two suffixes would invite a comparison between quantities with no relationship. The dashboard
 * already refuses to ratio them — its "share" is a count of running engines — so the units are free
 * to differ; what must not happen is one function silently relabelling both.
 */
export function formatPearlRate(thPerSecond: number | null | undefined): string {
  // `!(x > 0)` rather than `x <= 0` so a NaN reading fails closed as 0 instead of printing "NaN".
  if (thPerSecond === null || thPerSecond === undefined || !(thPerSecond > 0)) {
    return "0 TH/s";
  }
  if (thPerSecond >= 100) {
    return `${thPerSecond.toFixed(1)} TH/s`;
  }
  return `${thPerSecond.toFixed(2)} TH/s`;
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString();
}
