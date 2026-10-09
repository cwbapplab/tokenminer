/** Close-to-tray: the window-close behaviour and its setting. */

import { invokeCommand, isTauri } from "./tauri";

/**
 * Mirrors the close-to-tray preference into the Rust backend.
 *
 * The close handler runs in Rust and has to decide *before* the webview is torn
 * down or hidden, so it cannot ask the frontend at that moment — the value is
 * pushed here whenever the setting is loaded or changed. A no-op outside Tauri.
 */
export function syncCloseToTray(enabled: boolean): Promise<void> {
  if (!isTauri()) {
    return Promise.resolve();
  }
  return invokeCommand<void>("set_close_to_tray", { enabled });
}
