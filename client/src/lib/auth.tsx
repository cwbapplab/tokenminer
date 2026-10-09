import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { api, ApiError } from "./api";
import type { AuthTokens, PendingActivation, UserProfile } from "./types";

export type AuthStatus = "loading" | "authenticated" | "anonymous";

export interface SignInOutcome {
  /** The credentials were right, but the account still needs its OTP activation. */
  requiresActivation: boolean;
}

interface AuthContextValue {
  status: AuthStatus;
  user: UserProfile | null;
  signIn: (email: string, password: string) => Promise<SignInOutcome>;
  signInWithGoogle: (idToken: string) => Promise<SignInOutcome>;
  requestOtp: (email: string) => Promise<void>;
  verifyOtp: (email: string, code: string) => Promise<void>;
  verifyEmail: (token: string) => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function isPendingActivation(value: unknown): value is PendingActivation {
  return typeof value === "object" && value !== null && "requiresActivation" in value;
}

/**
 * Whether the server rejected the session outright, as opposed to being unreachable
 * (status 0) or failing in some other way. Only an outright rejection should discard tokens.
 */
function isSessionRejection(error: unknown): boolean {
  return error instanceof ApiError && (error.status === 401 || error.status === 403);
}

/** Boot-time profile retries: enough to ride out the API still starting up, not so many
 *  that a genuinely dead backend keeps the app on the loading spinner. */
const RESTORE_ATTEMPTS = 3;
const RESTORE_RETRY_MS = 700;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [user, setUser] = useState<UserProfile | null>(null);

  const loadProfile = useCallback(async () => {
    const profile = await api.get<UserProfile>("/api/auth/me");
    setUser(profile);
    setStatus("authenticated");
  }, []);

  const applyTokensOrPending = useCallback(
    async (response: AuthTokens | PendingActivation): Promise<SignInOutcome> => {
      if (isPendingActivation(response)) {
        return { requiresActivation: true };
      }
      api.setTokens(response);
      await loadProfile();
      return { requiresActivation: false };
    },
    [loadProfile],
  );

  // A stored token is not proof of a live session, so the profile is what decides.
  useEffect(() => {
    let cancelled = false;

    async function restore() {
      if (!api.hasSession) {
        setStatus("anonymous");
        return;
      }

      // At launch the API may not be reachable yet (it starts alongside the app, or the
      // machine's network is still coming up). A few spaced retries keep that transient
      // failure from looking like a signed-out session.
      for (let attempt = 0; attempt < RESTORE_ATTEMPTS; attempt++) {
        try {
          const profile = await api.get<UserProfile>("/api/auth/me");
          if (cancelled) {
            return;
          }
          setUser(profile);
          setStatus("authenticated");
          return;
        } catch (error) {
          if (cancelled) {
            return;
          }

          // A rejection is final — retrying cannot help, and the tokens are spent.
          if (isSessionRejection(error)) {
            api.clearTokens();
            setStatus("anonymous");
            return;
          }

          if (attempt < RESTORE_ATTEMPTS - 1) {
            await delay(RESTORE_RETRY_MS * (attempt + 1));
            if (cancelled) {
              return;
            }
          }
        }
      }

      // Still failing for a transient reason: show the login screen but keep the stored
      // tokens, so the next launch can resume the session instead of asking for the
      // password because the network blinked.
      setStatus("anonymous");
    }

    void restore();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    api.setUnauthorizedHandler(() => {
      setUser(null);
      setStatus("anonymous");
    });
    return () => api.setUnauthorizedHandler(null);
  }, []);

  const signIn = useCallback(
    async (email: string, password: string) => {
      const response = await api.post<AuthTokens | PendingActivation>("/api/auth/login", { email, password });
      return applyTokensOrPending(response);
    },
    [applyTokensOrPending],
  );

  const signInWithGoogle = useCallback(
    async (idToken: string) => {
      const response = await api.post<AuthTokens | PendingActivation>("/api/auth/google", { idToken });
      return applyTokensOrPending(response);
    },
    [applyTokensOrPending],
  );

  const requestOtp = useCallback(async (email: string) => {
    await api.post<void>("/api/auth/otp/request", { email });
  }, []);

  const verifyOtp = useCallback(
    async (email: string, code: string) => {
      // Verifying the code activates the account and signs it in in one step.
      const tokens = await api.post<AuthTokens>("/api/auth/otp/verify", { email, code });
      api.setTokens(tokens);
      await loadProfile();
    },
    [loadProfile],
  );

  const verifyEmail = useCallback(
    async (token: string) => {
      await api.post<void>("/api/auth/verify-email", { token });
    },
    [],
  );

  const signOut = useCallback(async () => {
    const refreshToken = api.currentRefreshToken;

    try {
      if (refreshToken) {
        await api.post<void>("/api/auth/logout", { refreshToken });
      }
    } catch {
      // Signing out locally matters more than telling the server about it.
    } finally {
      api.clearTokens();
      setUser(null);
      setStatus("anonymous");
    }
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({ status, user, signIn, signInWithGoogle, requestOtp, verifyOtp, verifyEmail, signOut }),
    [status, user, signIn, signInWithGoogle, requestOtp, verifyOtp, verifyEmail, signOut],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside an AuthProvider.");
  }
  return context;
}
