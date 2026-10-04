import { useEffect, useState } from "react";
import { getMiners, onMinerStatus } from "../lib/miners";
import { isTauri } from "../lib/tauri";
import type { MinerKind, MinerStatus } from "../lib/types";

export function emptyStatus(kind: MinerKind): MinerStatus {
  return {
    kind,
    state: "stopped",
    hashrate: 0,
    cpuHashrate: 0,
    gpuHashrate: 0,
    activeJobs: 0,
    workers: 0,
    message: null,
    updatedAt: new Date().toISOString(),
  };
}

/** Live status for both engines, seeded from the backend and updated via events. */
export function useMiners(): Record<MinerKind, MinerStatus> {
  const [statuses, setStatuses] = useState<Record<MinerKind, MinerStatus>>({
    quantus: emptyStatus("quantus"),
    pearl: emptyStatus("pearl"),
  });

  useEffect(() => {
    if (!isTauri()) {
      return;
    }

    let disposed = false;
    let detach: (() => void) | undefined;

    void (async () => {
      try {
        const initial = await getMiners();
        if (!disposed) {
          setStatuses((current) => {
            const next = { ...current };
            for (const status of initial) {
              next[status.kind] = status;
            }
            return next;
          });
        }
      } catch {
        // Miner commands are not wired yet; keep the defaults.
      }

      const off = await onMinerStatus((status) => {
        if (disposed) {
          return;
        }
        setStatuses((current) => ({ ...current, [status.kind]: status }));
      });

      if (disposed) {
        off();
      } else {
        detach = off;
      }
    })();

    return () => {
      disposed = true;
      detach?.();
    };
  }, []);

  return statuses;
}
