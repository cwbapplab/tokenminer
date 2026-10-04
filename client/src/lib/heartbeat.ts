import { api } from "./api";
import { getHardwareId } from "./hardware";
import { loadSettings } from "./settings";
import type { MiningHeartbeatAck } from "./types";

/**
 * Liveness socket for `/ws/mining`. The access token must go on the query string
 * because browsers/webviews cannot set headers on a WebSocket handshake.
 */
const HEARTBEAT_INTERVAL_MS = 5000;

let socket: WebSocket | null = null;
let timer: number | null = null;

export function startHeartbeat(onAck?: (ack: MiningHeartbeatAck) => void): void {
  stopHeartbeat();

  const token = api.currentAccessToken;
  if (!token) {
    return;
  }

  const base = loadSettings().apiBaseUrl.replace(/\/$/, "").replace(/^http/, "ws");
  const url = `${base}/ws/mining?access_token=${encodeURIComponent(token)}`;

  try {
    socket = new WebSocket(url);
  } catch {
    socket = null;
    return;
  }

  socket.onmessage = (event) => {
    try {
      onAck?.(JSON.parse(event.data as string) as MiningHeartbeatAck);
    } catch {
      /* ignore malformed frames */
    }
  };

  const hardwareId = getHardwareId();
  timer = window.setInterval(() => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "heartbeat", hardwareId }));
    }
  }, HEARTBEAT_INTERVAL_MS);
}

export function stopHeartbeat(): void {
  if (timer !== null) {
    window.clearInterval(timer);
    timer = null;
  }
  if (socket) {
    socket.onmessage = null;
    socket.close();
    socket = null;
  }
}
