import { useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Pickaxe, ShieldCheck } from "lucide-react";
import { useAuth } from "../lib/auth";
import { ApiError } from "../lib/api";
import { Button } from "../components/ui";
import { useToast } from "../components/Toast";

type Step = "credentials" | "otp";

export function LoginPage() {
  const { signIn, requestOtp, verifyOtp } = useAuth();
  const toast = useToast();
  const navigate = useNavigate();

  const [step, setStep] = useState<Step>("credentials");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [remember, setRemember] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submitCredentials(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const outcome = await signIn(email, password);
      if (outcome.requiresActivation) {
        await requestOtp(email);
        setStep("otp");
      } else {
        navigate("/", { replace: true });
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Login failed. Check that the API is running.");
    } finally {
      setBusy(false);
    }
  }

  async function submitOtp(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await verifyOtp(email, code);
      navigate("/", { replace: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Verification failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-full items-center justify-center bg-[#f4f6fa] px-4 py-10 dark:bg-slate-950">
      <div className="w-full max-w-md">
        <div className="rounded-lg border border-slate-200/80 bg-white p-8 shadow-[0_6px_24px_rgba(15,23,42,0.08)] dark:border-slate-800 dark:bg-slate-900">
          <div className="flex flex-col items-center gap-2">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-brand-500 text-white">
              <Pickaxe className="h-6 w-6" />
            </div>
            <span className="text-base font-semibold text-slate-800 dark:text-slate-100">TokenMiner</span>
          </div>

          {step === "credentials" ? (
            <>
              <div className="mt-6 space-y-2">
                <SocialButton
                  label="Sign In with Google"
                  onClick={() => toast.push("info", "Google sign-in is not configured yet.")}
                />
              </div>

              <div className="my-6 flex items-center gap-3 text-xs text-slate-400">
                <span className="h-px flex-1 bg-slate-200 dark:bg-slate-700" />
                OR
                <span className="h-px flex-1 bg-slate-200 dark:bg-slate-700" />
              </div>

              <h4 className="mb-4 text-center text-[15px] font-medium text-slate-700 dark:text-slate-200">
                Login with your email
              </h4>

              <form className="space-y-4" onSubmit={submitCredentials}>
                <input
                  type="email"
                  className={INPUT_CLASS}
                  placeholder="Email Address"
                  value={email}
                  autoFocus
                  onChange={(e) => setEmail(e.currentTarget.value)}
                />
                <input
                  type="password"
                  className={INPUT_CLASS}
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.currentTarget.value)}
                />

                <div className="flex items-center justify-between">
                  <label className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
                    <input
                      type="checkbox"
                      className="h-4 w-4 rounded accent-brand-500"
                      checked={remember}
                      onChange={(e) => setRemember(e.currentTarget.checked)}
                    />
                    Remember me?
                  </label>
                  <a href="#!" className="text-sm text-brand-500 hover:underline">
                    Forgot Password?
                  </a>
                </div>

                {error ? <p className="text-xs text-red-500">{error}</p> : null}

                <Button type="submit" loading={busy} className="w-full">
                  Login
                </Button>
              </form>

              <div className="mt-6 flex items-center justify-between text-sm">
                <span className="text-slate-500 dark:text-slate-400">Don&apos;t have an Account?</span>
                <a href="#!" className="font-medium text-brand-500 hover:underline">
                  Create Account
                </a>
              </div>
            </>
          ) : (
            <>
              <div className="mt-6 flex items-start gap-2 rounded-md bg-brand-50 p-3 text-xs text-brand-700 dark:bg-brand-950 dark:text-brand-200">
                <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0" />
                <span>
                  We sent a one-time code to <span className="font-medium">{email}</span>. Enter it to activate this
                  device.
                </span>
              </div>

              <form className="mt-4 space-y-4" onSubmit={submitOtp}>
                <input
                  className={INPUT_CLASS}
                  placeholder="One-time code"
                  value={code}
                  autoFocus
                  onChange={(e) => setCode(e.currentTarget.value)}
                />
                {error ? <p className="text-xs text-red-500">{error}</p> : null}
                <Button type="submit" loading={busy} className="w-full">
                  Verify &amp; continue
                </Button>
                <button
                  type="button"
                  className="w-full text-center text-xs text-slate-400 hover:text-brand-500"
                  onClick={() => setStep("credentials")}
                >
                  Back to login
                </button>
              </form>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

const INPUT_CLASS =
  "h-11 w-full rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-700 placeholder:text-slate-400 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/20 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100";

function SocialButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex h-11 w-full items-center justify-center gap-2 rounded-md border border-slate-200 bg-slate-50 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
    >
      <GoogleIcon />
      <span>{label}</span>
    </button>
  );
}

function GoogleIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 24 24" aria-hidden="true">
      <path
        fill="#4285F4"
        d="M23.5 12.3c0-.8-.1-1.6-.2-2.3H12v4.5h6.5a5.6 5.6 0 0 1-2.4 3.7v3h3.9c2.3-2.1 3.5-5.2 3.5-8.9z"
      />
      <path
        fill="#34A853"
        d="M12 24c3.2 0 5.9-1.1 7.9-2.9l-3.9-3c-1.1.7-2.4 1.2-4 1.2-3.1 0-5.7-2.1-6.6-4.9H1.4v3.1A12 12 0 0 0 12 24z"
      />
      <path fill="#FBBC05" d="M5.4 14.4a7.2 7.2 0 0 1 0-4.6V6.7H1.4a12 12 0 0 0 0 10.8l4-3.1z" />
      <path
        fill="#EA4335"
        d="M12 4.8c1.8 0 3.3.6 4.6 1.8l3.4-3.4A12 12 0 0 0 1.4 6.7l4 3.1C6.3 6.9 8.9 4.8 12 4.8z"
      />
    </svg>
  );
}
