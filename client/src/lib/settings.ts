/** Persisted, non-secret application settings (miner + API configuration). */

import type { MinerKind } from "./types";

export interface QuantusSettings {
  nodeAddr: string;
  authTokenFile: string;
  tlsCertSha256File: string;
  cpuWorkers: number;
  gpuDevices: number;
  cudaGpu: boolean;
  allowIntegrated: boolean;
  metricsPort: number;
}

export interface PearlSettings {
  certVersion: number;
}

export interface AppSettings {
  apiBaseUrl: string;
  /**
   * The Stratum endpoint the client connects to (the TokenMiner proxy). Takes
   * precedence over whatever host the API's launch command mentions, because a
   * template may reference the pool's base URL instead of `{stratumHost}`.
   */
  stratumEndpoint: string;
  /**
   * Which engine a session starts when the API does not name one. A fallback, not an
   * override: a session whose coin or algorithm says PRL/pearl still runs Pearl and one
   * that says QTC/quantus still runs Quantus, whatever this is set to. It only decides the
   * ambiguous case, which used to be hard-wired to Quantus.
   */
  defaultEngine: MinerKind;
  quantus: QuantusSettings;
  pearl: PearlSettings;
}

const STORAGE_KEY = "tokenminer.settings";

export const DEFAULT_SETTINGS: AppSettings = {
  // The dev stack binds 0.0.0.0, so the API and the proxy answer on the host's LAN address. These
  // must match the interface holding the default route on the machine running the stack.
  apiBaseUrl: "http://192.168.1.2:5210",
  stratumEndpoint: "192.168.1.2:3333",
  defaultEngine: "pearl",
  quantus: {
    nodeAddr: "127.0.0.1:9833",
    authTokenFile: "",
    tlsCertSha256File: "",
    cpuWorkers: 0,
    gpuDevices: 0,
    cudaGpu: false,
    allowIntegrated: false,
    metricsPort: 9900,
  },
  pearl: {
    // The certificate version picks how the commitment roots are salted before the noise seeds
    // are derived. It has to match what the pool sends in `mining.notify`; 1 and 2 are `Legacy`,
    // 3 is `Salted`.
    certVersion: 2,
  },
};

function isMinerKind(value: unknown): value is MinerKind {
  return value === "pearl" || value === "quantus";
}

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return DEFAULT_SETTINGS;
    }
    const parsed = JSON.parse(raw) as Partial<AppSettings>;
    return {
      ...DEFAULT_SETTINGS,
      ...parsed,
      // A stored engine name is checked rather than trusted: an invalid one would otherwise survive
      // the merge and fail at session start as a Rust deserialization error, which reads as a broken
      // session rather than a bad setting.
      defaultEngine: isMinerKind(parsed.defaultEngine) ? parsed.defaultEngine : DEFAULT_SETTINGS.defaultEngine,
      quantus: { ...DEFAULT_SETTINGS.quantus, ...parsed.quantus },
      pearl: { ...DEFAULT_SETTINGS.pearl, ...parsed.pearl },
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(settings: AppSettings): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}
