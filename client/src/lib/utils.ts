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

export function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString();
}
