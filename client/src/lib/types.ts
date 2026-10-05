/**
 * Wire types for the TokenMiner API.
 *
 * Mirrors the records in TokenMiner.Contracts. ASP.NET serialises with the default
 * camelCase policy, so the field names here match the JSON on the wire.
 */

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

export interface PendingActivation {
  requiresActivation: boolean;
}

export interface UserProfile {
  id: string;
  email: string;
  displayName: string | null;
  status: string;
  emailConfirmed: boolean;
  roles: string[];
  createdAt: string;
}

// --- Mining session -------------------------------------------------------------------------

export interface StartMiningRequest {
  hardwareId: string;
  poolId?: string | null;
}

/** The algorithm configuration the API rendered (placeholders already substituted). */
export interface MinerConfig {
  algo?: string;
  /** The proxy host:port the client must connect to. */
  endpoint?: string;
  wallet?: string;
  workerId?: string;
  coin?: string;
  supportedCoins?: string[];
  [key: string]: unknown;
}

export interface MiningSession {
  sessionId: string;
  poolId: string;
  poolName: string;
  coinId: string;
  coinCode: string;
  algorithmId: string;
  algorithmCode: string;
  workerId: string;
  minerCommand: string;
  /** Structured config — the form the client reads directly. */
  minerConfig: MinerConfig;
  /** The proxy host:port the client must connect to (never the pool). */
  stratumEndpoint: string;
  status: string;
  startedAt: string;
}

export type StopMiningReason =
  | "connection-failure"
  | "driver-crash"
  | "miner-closed"
  | "user-triggered"
  | "unknown";

export interface StopMiningRequest {
  hardwareId: string;
  reason: StopMiningReason;
}

export interface MiningHeartbeatAck {
  type: string;
  sessionFound: boolean;
  status: string | null;
  resumed: boolean;
  at: string;
}

// --- Analytics ------------------------------------------------------------------------------

export interface HardwareAnalytics {
  userHardwareId: string;
  last30Usd: number;
  last60Usd: number;
  allTimeUsd: number;
}

export interface AnalyticsTotals {
  last30Usd: number;
  last60Usd: number;
  allTimeUsd: number;
}

export interface MiningAnalytics {
  hardware: HardwareAnalytics[];
  totals: AnalyticsTotals;
}

// --- Local miner status (produced by our Rust backend, not the API) -------------------------

export type MinerKind = "quantus" | "pearl";

export type MinerState = "stopped" | "starting" | "running" | "error";

export interface MinerStatus {
  kind: MinerKind;
  state: MinerState;
  /**
   * Rate across the engine's workers, in the unit that engine reports: H/s for Quantus, TH/s for
   * Pearl. Not a common unit — a Pearl TH is 10^12 int8 MACs, not 10^12 Quantus hashes — so format
   * with `formatPearlRate` or `formatHashrate` by `kind`, never with one function for both.
   */
  hashrate: number;
  /** Same per-engine unit as `hashrate`. */
  cpuHashrate: number;
  gpuHashrate: number;
  activeJobs: number;
  workers: number;
  message: string | null;
  updatedAt: string;
}

/** Parameters parsed out of the API's rendered miner command. */
export interface MinerCommandParams {
  program: string;
  algo: string | null;
  host: string | null;
  wallet: string | null;
  worker: string | null;
  cpuWorkers: number | null;
  gpuDevices: number | null;
  args: string[];
}

export interface SessionStarted {
  kind: MinerKind;
  params: MinerCommandParams;
  status: MinerStatus;
}
