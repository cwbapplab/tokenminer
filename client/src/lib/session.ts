import { api } from "./api";
import { getHardwareId } from "./hardware";
import type { MiningAnalytics, MiningSession, StopMiningReason } from "./types";

/** Opens a mining session for this device and returns the backend's session details. */
export function startSession(poolId?: string | null): Promise<MiningSession> {
  return api.post<MiningSession>("/api/mining/start", {
    hardwareId: getHardwareId(),
    poolId: poolId ?? null,
  });
}

/** Closes the device's mining session. */
export function stopSession(reason: StopMiningReason): Promise<void> {
  return api.post<void>("/api/mining/stop", {
    hardwareId: getHardwareId(),
    reason,
  });
}

/**
 * The device's in-flight session, if the API still holds one. The API returns 204 when there
 * is none, which the client surfaces as `null`.
 */
export async function getSession(): Promise<MiningSession | null> {
  const session = await api.get<MiningSession | undefined>(
    `/api/mining/session?hardwareId=${encodeURIComponent(getHardwareId())}`,
  );
  return session ?? null;
}

/** Per-device and total USD earnings. */
export function getAnalytics(userHardwareId?: string): Promise<MiningAnalytics> {
  const suffix = userHardwareId ? `?userHardwareId=${encodeURIComponent(userHardwareId)}` : "";
  return api.get<MiningAnalytics>(`/api/analytics${suffix}`);
}
