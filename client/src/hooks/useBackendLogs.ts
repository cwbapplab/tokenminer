import { useEffect, useState } from "react";
import { attachLogger } from "@tauri-apps/plugin-log";
import { isTauri } from "../lib/tauri";
import type { LogLine } from "../components/LogConsole";

/** Strips the ANSI colour codes the Rust logger embeds in its lines. */
const ANSI = /\u001b\[[0-9;]*m/g;

/**
 * Streams the Rust `log` crate output into the UI.
 *
 * Requires the `Webview` target on `tauri-plugin-log` in `lib.rs`; without it the
 * backend only writes to stdout/the log file and nothing arrives here.
 */
export function useBackendLogs(limit = 400): LogLine[] {
  const [lines, setLines] = useState<LogLine[]>([]);

  useEffect(() => {
    if (!isTauri()) {
      return;
    }

    let disposed = false;
    let detach: (() => void) | undefined;

    void (async () => {
      const off = await attachLogger(({ level, message }) => {
        if (disposed) {
          return;
        }
        setLines((current) =>
          [
            ...current,
            {
              id: Date.now() + Math.random(),
              level: String(level).toLowerCase(),
              message: message.replace(ANSI, "").trimEnd(),
              timestamp: "",
            },
          ].slice(-limit),
        );
      });

      // If the effect was already cleaned up (StrictMode remount), detach now so we
      // never leave a second listener attached.
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
  }, [limit]);

  return lines;
}
