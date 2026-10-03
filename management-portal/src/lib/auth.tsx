import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { api } from './api';
import type { AuthTokens, PendingActivation, UserProfile } from './types';

export type AuthStatus = 'loading' | 'authenticated' | 'anonymous';

export interface SignInOutcome {
  /** The credentials were right, but the account still needs its OTP activation. */
  requiresActivation: boolean;
}

interface AuthContextValue {
  status: AuthStatus;
  user: UserProfile | null;
  isAdmin: boolean;
  signIn: (email: string, password: string) => Promise<SignInOutcome>;
  requestOtp: (email: string) => Promise<void>;
  verifyOtp: (email: string, code: string) => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function isPendingActivation(value: unknown): value is PendingActivation {
  return typeof value === 'object' && value !== null && 'requiresActivation' in value;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>('loading');
  const [user, setUser] = useState<UserProfile | null>(null);

  const loadProfile = useCallback(async () => {
    const profile = await api.get<UserProfile>('/api/auth/me');
    setUser(profile);
    setStatus('authenticated');
  }, []);

  // A stored token is not proof of a live session, so the profile is what decides.
  useEffect(() => {
    let cancelled = false;

    async function restore() {
      if (!api.hasSession) {
        setStatus('anonymous');
        return;
      }

      try {
        const profile = await api.get<UserProfile>('/api/auth/me');

        if (!cancelled) {
          setUser(profile);
          setStatus('authenticated');
        }
      } catch {
        if (!cancelled) {
          api.clearTokens();
          setStatus('anonymous');
        }
      }
    }

    void restore();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    api.setUnauthorizedHandler(() => {
      setUser(null);
      setStatus('anonymous');
    });

    return () => api.setUnauthorizedHandler(null);
  }, []);

  const signIn = useCallback(
    async (email: string, password: string): Promise<SignInOutcome> => {
      const response = await api.post<AuthTokens | PendingActivation>('/api/auth/login', {
        email,
        password,
      });

      if (isPendingActivation(response)) {
        return { requiresActivation: true };
      }

      api.setTokens(response);
      await loadProfile();

      return { requiresActivation: false };
    },
    [loadProfile],
  );

  const requestOtp = useCallback(async (email: string) => {
    await api.post<void>('/api/auth/otp/request', { email });
  }, []);

  const verifyOtp = useCallback(
    async (email: string, code: string) => {
      // Verifying the code activates the account and signs it in in one step.
      const tokens = await api.post<AuthTokens>('/api/auth/otp/verify', { email, code });
      api.setTokens(tokens);
      await loadProfile();
    },
    [loadProfile],
  );

  const signOut = useCallback(async () => {
    const refreshToken = api.currentRefreshToken;

    try {
      if (refreshToken) {
        await api.post<void>('/api/auth/logout', { refreshToken });
      }
    } catch {
      // Signing out locally matters more than telling the server about it.
    } finally {
      api.clearTokens();
      setUser(null);
      setStatus('anonymous');
    }
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      status,
      user,
      isAdmin: user?.roles.includes('admin') ?? false,
      signIn,
      requestOtp,
      verifyOtp,
      signOut,
    }),
    [status, user, signIn, requestOtp, verifyOtp, signOut],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error('useAuth must be used inside an AuthProvider.');
  }

  return context;
}
