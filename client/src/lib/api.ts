import { httpFetch } from "./tauri";
import { loadSettings } from "./settings";
import type { AuthTokens } from "./types";

/** Raised for any non-success response, carrying whatever the API said about it. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string | null;

  constructor(message: string, status: number, detail: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/**
 * A persisted token pair, plus the wall-clock instant the access token stops being accepted.
 * The absolute instant is what lets the client refresh ahead of expiry across a reload; the
 * relative `expiresIn` alone cannot be trusted once any time has passed.
 */
export type StoredTokens = AuthTokens & { expiresAt: number };

/**
 * Where the access/refresh tokens live. The default is localStorage; in the desktop app
 * this is swapped for an OS-encrypted store once the Rust side is available.
 */
export interface TokenStorage {
  load(): StoredTokens | null;
  save(tokens: AuthTokens): void;
  clear(): void;
}

const ACCESS_KEY = "tokenminer.tokens.access";
const REFRESH_KEY = "tokenminer.tokens.refresh";
const EXPIRES_KEY = "tokenminer.tokens.expiresIn";
const EXPIRES_AT_KEY = "tokenminer.tokens.expiresAt";

export const localStorageTokenStorage: TokenStorage = {
  load() {
    const accessToken = localStorage.getItem(ACCESS_KEY);
    const refreshToken = localStorage.getItem(REFRESH_KEY);
    if (!accessToken || !refreshToken) {
      return null;
    }
    const expiresIn = Number(localStorage.getItem(EXPIRES_KEY) ?? "0");
    const storedExpiry = Number(localStorage.getItem(EXPIRES_AT_KEY) ?? "0");
    return {
      accessToken,
      refreshToken,
      expiresIn,
      // Sessions written before the absolute timestamp existed fall back to counting from
      // now, rather than pinning the expiry at the epoch and forcing a spurious refresh.
      expiresAt: storedExpiry > 0 ? storedExpiry : Date.now() + expiresIn * 1000,
    };
  },
  save(tokens) {
    localStorage.setItem(ACCESS_KEY, tokens.accessToken);
    localStorage.setItem(REFRESH_KEY, tokens.refreshToken);
    localStorage.setItem(EXPIRES_KEY, String(tokens.expiresIn));
    localStorage.setItem(EXPIRES_AT_KEY, String(Date.now() + tokens.expiresIn * 1000));
  },
  clear() {
    localStorage.removeItem(ACCESS_KEY);
    localStorage.removeItem(REFRESH_KEY);
    localStorage.removeItem(EXPIRES_KEY);
    localStorage.removeItem(EXPIRES_AT_KEY);
  },
};

/** Refresh this long before the access token expires, so a request never races the expiry. */
const REFRESH_LEAD_MS = 60_000;
/** Floor for the schedule: an already-expired token refreshes promptly, but not in a hot loop. */
const MIN_REFRESH_DELAY_MS = 5_000;
/** Backoff before retrying a refresh that could not reach the server. */
const REFRESH_RETRY_MS = 30_000;

type UnauthorizedHandler = () => void;

/**
 * Thin wrapper over the API.
 *
 * Owns two things callers should not have to think about: attaching the bearer token,
 * and recovering from an expired access token by refreshing once and replaying the request.
 */
class ApiClient {
  private storage: TokenStorage = localStorageTokenStorage;
  private accessToken: string | null = null;
  private refreshToken: string | null = null;
  private accessExpiresAt: number | null = null;
  private refreshInFlight: Promise<RefreshOutcome> | null = null;
  private refreshTimer: number | null = null;
  private onUnauthorized: UnauthorizedHandler | null = null;

  constructor() {
    this.hydrate();
  }

  setTokenStorage(storage: TokenStorage): void {
    this.storage = storage;
    this.hydrate();
  }

  private hydrate(): void {
    const tokens = this.storage.load();
    this.accessToken = tokens?.accessToken ?? null;
    this.refreshToken = tokens?.refreshToken ?? null;
    this.accessExpiresAt = tokens?.expiresAt ?? null;
    this.scheduleProactiveRefresh();
  }

  get hasSession(): boolean {
    return this.accessToken !== null;
  }

  get currentRefreshToken(): string | null {
    return this.refreshToken;
  }

  /** The current access token, for the mining WebSocket (query-string auth). */
  get currentAccessToken(): string | null {
    return this.accessToken;
  }

  /** Called when a refresh fails, so the app can send the user back to the sign-in page. */
  setUnauthorizedHandler(handler: UnauthorizedHandler | null): void {
    this.onUnauthorized = handler;
  }

  setTokens(tokens: AuthTokens): void {
    this.accessToken = tokens.accessToken;
    this.refreshToken = tokens.refreshToken;
    this.accessExpiresAt = Date.now() + tokens.expiresIn * 1000;
    this.storage.save(tokens);
    this.scheduleProactiveRefresh();
  }

  clearTokens(): void {
    this.accessToken = null;
    this.refreshToken = null;
    this.accessExpiresAt = null;
    this.clearRefreshTimer();
    this.storage.clear();
  }

  /**
   * Refreshes the access token if it is missing or close to expiring. Safe to call from any
   * user-visible moment (window focus, wake from sleep, visibility change): it is a no-op
   * while the token is still comfortable, and callers racing it share one refresh.
   */
  async ensureFreshAccessToken(): Promise<void> {
    if (this.refreshToken === null) {
      return;
    }

    const comfortable =
      this.accessToken !== null &&
      this.accessExpiresAt !== null &&
      this.accessExpiresAt - Date.now() > REFRESH_LEAD_MS;
    if (comfortable) {
      return;
    }

    const outcome = await this.refresh();

    if (outcome === "rejected") {
      this.clearTokens();
      this.onUnauthorized?.();
    } else if (outcome === "unreachable") {
      // Keep the session and retry shortly, rather than leaving the client with a dead token.
      this.scheduleRefresh(REFRESH_RETRY_MS);
    }
  }

  /**
   * Refreshes ahead of the access token's expiry so a session that merely went idle does not
   * have to be rescued by a rejected request. The timer is throttled while the window is
   * hidden or the machine sleeps, which is what `ensureFreshAccessToken` covers on resume.
   */
  private scheduleProactiveRefresh(): void {
    if (this.refreshToken === null || this.accessExpiresAt === null) {
      this.clearRefreshTimer();
      return;
    }

    const delayMs = Math.max(this.accessExpiresAt - Date.now() - REFRESH_LEAD_MS, MIN_REFRESH_DELAY_MS);
    this.scheduleRefresh(delayMs);
  }

  private scheduleRefresh(delayMs: number): void {
    this.clearRefreshTimer();
    this.refreshTimer = window.setTimeout(() => {
      void this.ensureFreshAccessToken();
    }, delayMs);
  }

  private clearRefreshTimer(): void {
    if (this.refreshTimer !== null) {
      window.clearTimeout(this.refreshTimer);
      this.refreshTimer = null;
    }
  }

  get<T>(path: string): Promise<T> {
    return this.request<T>(path, { method: "GET" });
  }

  post<T>(path: string, body?: unknown): Promise<T> {
    return this.request<T>(path, { method: "POST", body: serialize(body) });
  }

  put<T>(path: string, body?: unknown): Promise<T> {
    return this.request<T>(path, { method: "PUT", body: serialize(body) });
  }

  delete<T>(path: string): Promise<T> {
    return this.request<T>(path, { method: "DELETE" });
  }

  async request<T>(path: string, init: RequestInit, allowRetry = true): Promise<T> {
    const headers = new Headers(init.headers);
    headers.set("Accept", "application/json");

    if (init.body !== undefined && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }

    if (this.accessToken) {
      headers.set("Authorization", `Bearer ${this.accessToken}`);
    }

    const url = `${loadSettings().apiBaseUrl.replace(/\/$/, "")}${path}`;
    const response = await httpFetch(url, { ...init, headers });

    if (response.status === 401 && allowRetry && this.refreshToken) {
      const outcome = await this.refresh();

      if (outcome === "ok") {
        return this.request<T>(path, init, false);
      }

      // The server could not be reached to refresh. The session is probably fine, so it is
      // kept and the caller just sees a failure it can retry — signing the user out here is
      // what turns a flaky moment into a trip back to the login screen.
      if (outcome === "unreachable") {
        throw new ApiError("Cannot reach the server. Check your connection and try again.", 0, null);
      }
    }

    if (!response.ok) {
      const problem = await readProblem(response);

      if (response.status === 401) {
        // Reaching here means the refresh token was rejected, not merely unreadable.
        this.clearTokens();
        this.onUnauthorized?.();
      }

      throw new ApiError(problem.title, response.status, problem.detail);
    }

    return parseBody<T>(response);
  }

  /** Collapses concurrent refreshes so a page load does not fire one per request. */
  private refresh(): Promise<RefreshOutcome> {
    if (this.refreshInFlight) {
      return this.refreshInFlight;
    }

    const token = this.refreshToken;
    if (!token) {
      return Promise.resolve("rejected");
    }

    const attempt = this.performRefresh(token).finally(() => {
      this.refreshInFlight = null;
    });

    this.refreshInFlight = attempt;
    return attempt;
  }

  private async performRefresh(token: string): Promise<RefreshOutcome> {
    let response: Response;
    try {
      const url = `${loadSettings().apiBaseUrl.replace(/\/$/, "")}/api/auth/refresh`;
      response = await httpFetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ refreshToken: token }),
      });
    } catch {
      // Network error / timeout: the token was never judged, so the session stands.
      return "unreachable";
    }

    if (!response.ok) {
      // A rejection is a verdict on the token; anything else (5xx, gateway errors) is the
      // server being unavailable and must not end the session.
      return isRejection(response.status) ? "rejected" : "unreachable";
    }

    try {
      this.setTokens((await response.json()) as AuthTokens);
      return "ok";
    } catch {
      return "unreachable";
    }
  }
}

/**
 * Whether a status means "this token is no longer accepted" (end the session) rather than
 * "the server could not answer" (keep it and retry later). Only an explicit authentication
 * rejection qualifies — a 5xx or a malformed-request 400 says nothing about the token.
 */
function isRejection(status: number): boolean {
  return status === 401 || status === 403;
}

/**
 * The result of a refresh attempt. `rejected` is a real answer from the server that the
 * refresh token is no longer valid; `unreachable` means no answer was obtained.
 */
type RefreshOutcome = "ok" | "rejected" | "unreachable";

function serialize(body: unknown): string | undefined {
  return body === undefined ? undefined : JSON.stringify(body);
}

async function parseBody<T>(response: Response): Promise<T> {
  if (response.status === 204) {
    return undefined as T;
  }

  const text = await response.text();
  if (text.length === 0) {
    return undefined as T;
  }

  return JSON.parse(text) as T;
}

interface ProblemLike {
  title?: string;
  detail?: string;
  errors?: Record<string, string[]>;
}

/** Turns an RFC 7807 problem response into something worth showing. */
async function readProblem(response: Response): Promise<{ title: string; detail: string | null }> {
  const fallback = `Request failed (${response.status} ${response.statusText})`;

  try {
    const text = await response.text();
    if (text.length === 0) {
      return { title: fallback, detail: null };
    }

    const problem = JSON.parse(text) as ProblemLike;
    const fieldErrors = flattenFieldErrors(problem.errors);
    const title = problem.title ?? fallback;

    return {
      title: fieldErrors ?? title,
      detail: fieldErrors ? null : problem.detail ?? null,
    };
  } catch {
    return { title: fallback, detail: null };
  }
}

function flattenFieldErrors(errors: Record<string, string[]> | undefined): string | null {
  if (!errors) {
    return null;
  }
  const messages = Object.values(errors).flat().filter(Boolean);
  return messages.length > 0 ? messages.join(" ") : null;
}

export const api = new ApiClient();

/** Builds a query string, omitting empty values. */
export function query(params: Record<string, string | number | boolean | undefined | null>): string {
  const search = new URLSearchParams();

  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, String(value));
    }
  }

  const result = search.toString();
  return result.length > 0 ? `?${result}` : "";
}
