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
 * Pearl's hashrate, in the only unit it has: candidate tiles per second.
 *
 * A Pearl hash is one jackpot digest over one 16×16 candidate tile — `compute_jackpot_hash` runs
 * once per tile and `check_jackpot_against_nbits` compares that one digest — so a tile *is* a hash
 * and the number needs no scaling. What it must not borrow is the "H/s" suffix: one of these hashes
 * is 16 × 16 × 4096 int8 MACs, which is not one of Quantus's, so the same suffix on both engines
 * invites a comparison between quantities that have no relationship.
 */
export function formatPearlRate(tilesPerSecond: number | null | undefined): string {
  if (tilesPerSecond === null || tilesPerSecond === undefined || tilesPerSecond <= 0) {
    return "0 tiles/s";
  }
  const units: Array<[number, string]> = [
    [1e9, "Gtiles/s"],
    [1e6, "Mtiles/s"],
    [1e3, "Ktiles/s"],
  ];
  for (const [scale, unit] of units) {
    if (tilesPerSecond >= scale) {
      return `${(tilesPerSecond / scale).toFixed(2)} ${unit}`;
    }
  }
  return `${tilesPerSecond.toFixed(2)} tiles/s`;
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString();
}
