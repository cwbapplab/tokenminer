import type { AuthTokens } from './types';

const ACCESS_TOKEN_KEY = 'tokenminer.portal.accessToken';
const REFRESH_TOKEN_KEY = 'tokenminer.portal.refreshToken';

/** Raised for any non-success response, carrying whatever the API said about it. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string | null;

  constructor(message: string, status: number, detail: string | null = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

type UnauthorizedHandler = () => void;

/**
 * Thin wrapper over fetch.
 *
 * Two things it owns that callers should not have to think about: attaching the bearer token,
 * and recovering from an expired access token by refreshing once and replaying the request.
 */
class ApiClient {
  private accessToken: string | null = null;
  private refreshToken: string | null = null;
  private refreshInFlight: Promise<boolean> | null = null;
  private onUnauthorized: UnauthorizedHandler | null = null;

  constructor() {
    this.accessToken = localStorage.getItem(ACCESS_TOKEN_KEY);
    this.refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);
  }

  get hasSession(): boolean {
    return this.accessToken !== null;
  }

  get currentRefreshToken(): string | null {
    return this.refreshToken;
  }

  /** Called when a refresh fails, so the app can send the operator back to the sign-in page. */
  setUnauthorizedHandler(handler: UnauthorizedHandler | null): void {
    this.onUnauthorized = handler;
  }

  setTokens(tokens: AuthTokens): void {
    this.accessToken = tokens.accessToken;
    this.refreshToken = tokens.refreshToken;
    localStorage.setItem(ACCESS_TOKEN_KEY, tokens.accessToken);
    localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refreshToken);
  }

  clearTokens(): void {
    this.accessToken = null;
    this.refreshToken = null;
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
  }

  get<T>(path: string): Promise<T> {
    return this.request<T>(path, { method: 'GET' });
  }

  post<T>(path: string, body?: unknown): Promise<T> {
    return this.request<T>(path, { method: 'POST', body: serialize(body) });
  }

  put<T>(path: string, body?: unknown): Promise<T> {
    return this.request<T>(path, { method: 'PUT', body: serialize(body) });
  }

  delete<T>(path: string): Promise<T> {
    return this.request<T>(path, { method: 'DELETE' });
  }

  async request<T>(path: string, init: RequestInit, allowRetry = true): Promise<T> {
    const headers = new Headers(init.headers);
    headers.set('Accept', 'application/json');

    if (init.body !== undefined && !headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json');
    }

    if (this.accessToken) {
      headers.set('Authorization', `Bearer ${this.accessToken}`);
    }

    const response = await fetch(path, { ...init, headers });

    if (response.status === 401 && allowRetry && this.refreshToken) {
      // One attempt only: a second 401 means the session is genuinely over.
      const refreshed = await this.refresh();

      if (refreshed) {
        return this.request<T>(path, init, false);
      }
    }

    if (!response.ok) {
      const problem = await readProblem(response);

      if (response.status === 401) {
        this.clearTokens();
        this.onUnauthorized?.();
      }

      throw new ApiError(problem.title, response.status, problem.detail);
    }

    return parseBody<T>(response);
  }

  /** Collapses concurrent refreshes so a page load does not fire one per request. */
  private refresh(): Promise<boolean> {
    if (this.refreshInFlight) {
      return this.refreshInFlight;
    }

    const token = this.refreshToken;

    if (!token) {
      return Promise.resolve(false);
    }

    this.refreshInFlight = (async () => {
      try {
        const response = await fetch('/api/auth/refresh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify({ refreshToken: token }),
        });

        if (!response.ok) {
          return false;
        }

        this.setTokens((await response.json()) as AuthTokens);
        return true;
      } catch {
        return false;
      } finally {
        this.refreshInFlight = null;
      }
    })();

    return this.refreshInFlight;
  }
}

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

/** Turns an RFC 7807 problem response into something worth showing an operator. */
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

  return messages.length > 0 ? messages.join(' ') : null;
}

export const api = new ApiClient();

/** Builds a query string, omitting empty values. */
export function query(params: Record<string, string | number | boolean | undefined | null>): string {
  const search = new URLSearchParams();

  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      search.set(key, String(value));
    }
  }

  const result = search.toString();

  return result.length > 0 ? `?${result}` : '';
}
