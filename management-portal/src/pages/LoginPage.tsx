import { useState } from 'react';
import type { FormEvent } from 'react';
import { Navigate, useLocation, useNavigate } from 'react-router-dom';
import { Icon } from '../components/icons';
import { useAuth } from '../lib/auth';

type Step = 'credentials' | 'activation';

export function LoginPage() {
  const { status, isAdmin, signIn, requestOtp, verifyOtp, signOut } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [step, setStep] = useState<Step>('credentials');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [code, setCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const redirectTo = (location.state as { from?: string } | null)?.from ?? '/';

  if (status === 'authenticated' && isAdmin) {
    return <Navigate to={redirectTo} replace />;
  }

  async function handleSignIn(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);

    try {
      const outcome = await signIn(email.trim(), password);

      if (outcome.requiresActivation) {
        setStep('activation');
        setNotice('This account still needs activating. Sending a one-time code to your email…');

        try {
          await requestOtp(email.trim());
          setNotice('Enter the code we just emailed you to activate the account.');
        } catch {
          setNotice(null);
          setError('Could not send the activation code. Try again in a moment.');
        }

        return;
      }

      navigate(redirectTo, { replace: true });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Sign-in failed.');
    } finally {
      setBusy(false);
    }
  }

  async function handleVerify(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);

    try {
      await verifyOtp(email.trim(), code.trim());
      navigate(redirectTo, { replace: true });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'That code was not accepted.');
    } finally {
      setBusy(false);
    }
  }

  async function resend() {
    setBusy(true);
    setError(null);

    try {
      await requestOtp(email.trim());
      setNotice('A new code is on its way.');
    } catch {
      setError('Could not send another code yet. Wait a moment and try again.');
    } finally {
      setBusy(false);
    }
  }

  // Signed in but without the admin role: the portal has nothing to show them.
  if (status === 'authenticated' && !isAdmin) {
    return (
      <div className="login-page">
        <div className="login-card">
          <div className="login-brand">
            <span className="brand-mark">T</span>
            <h3>Not an administrator</h3>
            <p>
              <strong>{email}</strong> is signed in but does not hold the <code>admin</code> role.
            </p>
          </div>

          <div className="alert alert-warning">
            Ask an operator to grant the role, or add the address to <code>Auth:BootstrapAdminEmails</code> and
            restart the API.
          </div>

          <button
            type="button"
            className="btn btn-primary w-100"
            onClick={async () => {
              await signOut();
            }}
          >
            <Icon name="logout" />
            Sign out
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-brand">
          <span className="brand-mark">T</span>
          <h3 className="mb-0">TokenMiner</h3>
          <p>Management portal &middot; operators only</p>
        </div>

        {error && <div className="alert alert-danger mb-3">{error}</div>}
        {notice && !error && <div className="alert alert-info mb-3">{notice}</div>}

        {step === 'credentials' && (
          <form onSubmit={handleSignIn}>
            <div className="mb-3">
              <label className="form-label" htmlFor="email">
                Email
              </label>
              <input
                id="email"
                className="form-control"
                type="email"
                autoComplete="username"
                required
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="operator@example.com"
              />
            </div>

            <div className="mb-4">
              <label className="form-label" htmlFor="password">
                Password
              </label>
              <input
                id="password"
                className="form-control"
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="••••••••••••"
              />
            </div>

            <button type="submit" className="btn btn-primary w-100" disabled={busy}>
              <Icon name="lock" />
              {busy ? 'Signing in…' : 'Sign in'}
            </button>
          </form>
        )}

        {step === 'activation' && (
          <form onSubmit={handleVerify}>
            <div className="mb-3">
              <label className="form-label" htmlFor="code">
                One-time code
              </label>
              <input
                id="code"
                className="form-control"
                inputMode="numeric"
                autoComplete="one-time-code"
                required
                value={code}
                onChange={(event) => setCode(event.target.value)}
                placeholder="123456"
              />
              <div className="field-hint">
                Activating <strong>{email}</strong>. Verifying the code signs you in.
              </div>
            </div>

            <button type="submit" className="btn btn-primary w-100 mb-2" disabled={busy}>
              <Icon name="check" />
              {busy ? 'Verifying…' : 'Activate and sign in'}
            </button>

            <div className="d-flex gap-2">
              <button type="button" className="btn btn-outline-secondary w-100" onClick={resend} disabled={busy}>
                <Icon name="refresh" />
                Resend code
              </button>
              <button
                type="button"
                className="btn btn-outline-secondary w-100"
                onClick={() => {
                  setStep('credentials');
                  setCode('');
                  setError(null);
                  setNotice(null);
                }}
                disabled={busy}
              >
                Back
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
