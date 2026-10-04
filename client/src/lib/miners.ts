import { invokeCommand, listenEvent } from "./tauri";
import type { MinerKind, MinerStatus, SessionStarted } from "./types";

export interface MinerLogLine {
  kind: MinerKind;
  level: string;
  message: string;
  timestamp: string;
}

/** Starts a local miner engine (Quantus or Pearl) in-process. */
export function startMiner(kind: MinerKind, config?: unknown): Promise<MinerStatus> {
  return invokeCommand<MinerStatus>("start_miner", { kind, config });
}

/**
 * Starts a mining session: the Rust side parses the API's command, picks the
 * engine, and runs it in-process (no child process).
 */
export function startSessionMiner(config: unknown): Promise<SessionStarted> {
  return invokeCommand<SessionStarted>("start_session_miner", { config });
}

/** Stops a local miner engine. */
export function stopMiner(kind: MinerKind): Promise<void> {
  return invokeCommand<void>("stop_miner", { kind });
}

/** Current status of every engine. */
export function getMiners(): Promise<MinerStatus[]> {
  return invokeCommand<MinerStatus[]>("get_miner_status");
}

/** Subscribes to live status updates emitted by the Rust backend. */
export function onMinerStatus(handler: (status: MinerStatus) => void) {
  return listenEvent<MinerStatus>("miner-status", handler);
}

/** Subscribes to live miner log lines emitted by the Rust backend. */
export function onMinerLog(handler: (line: MinerLogLine) => void) {
  return listenEvent<MinerLogLine>("miner-log", handler);
}
