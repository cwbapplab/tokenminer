import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { fetch as tauriFetch } from "@tauri-apps/plugin-http";

/**
 * Small bridge over the Tauri APIs so the React app can also run in a plain
 * browser (feature-detected) — used for future web/mobile targets.
 */
export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/** Invokes a Rust command. Only valid inside Tauri. */
export function invokeCommand<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  return invoke<T>(command, args);
}

/** Subscribes to a Rust-emitted event. Only valid inside Tauri. */
export function listenEvent<T>(event: string, handler: (payload: T) => void): Promise<UnlistenFn> {
  return listen<T>(event, (e) => handler(e.payload));
}

/**
 * HTTP through the Rust side when running in Tauri (reqwest — no webview CORS),
 * falling back to the browser fetch otherwise.
 */
export function httpFetch(input: string, init?: RequestInit): Promise<Response> {
  if (isTauri()) {
    return tauriFetch(input, init) as unknown as Promise<Response>;
  }
  return fetch(input, init);
}
