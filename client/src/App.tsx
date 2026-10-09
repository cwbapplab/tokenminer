import { useEffect } from "react";
import type { ReactNode } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Spinner } from "./components/ui";
import { ThemeProvider } from "./components/theme";
import { ToastProvider, useToast } from "./components/Toast";
import { AuthProvider, useAuth } from "./lib/auth";
import { loadSettings } from "./lib/settings";
import { isTauri } from "./lib/tauri";
import { syncCloseToTray } from "./lib/window";
import { LoginPage } from "./pages/LoginPage";
import { DashboardPage } from "./pages/DashboardPage";
import { MinersPage } from "./pages/MinersPage";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { SettingsPage } from "./pages/SettingsPage";

/** Routes `tokenminer://verify-email?token=…` deep links into the API. */
function DeepLinkHandler() {
  const { verifyEmail } = useAuth();
  const toast = useToast();

  useEffect(() => {
    if (!isTauri()) {
      return;
    }

    let disposed = false;
    let detach: (() => void) | undefined;

    async function handle(urls: string[] | null) {
      for (const url of urls ?? []) {
        try {
          const parsed = new URL(url);
          if (parsed.protocol !== "tokenminer:" || parsed.host !== "verify-email") {
            continue;
          }
          const token = parsed.searchParams.get("token");
          if (!token) {
            continue;
          }
          await verifyEmail(token);
          toast.push("success", "Email verified.");
        } catch {
          toast.push("error", "That verification link could not be processed.");
        }
      }
    }

    void (async () => {
      const { getCurrent, onOpenUrl } = await import("@tauri-apps/plugin-deep-link");
      if (disposed) {
        return;
      }
      await handle(await getCurrent());

      const off = await onOpenUrl((urls) => void handle(urls));
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
  }, [verifyEmail, toast]);

  return null;
}

/**
 * Pushes the persisted close-to-tray setting into the Rust backend once at boot.
 *
 * The backend defaults to enabled; if the user turned it off in a previous run,
 * this is what makes the window close for real again. It runs regardless of auth
 * so the setting takes effect even on the login screen.
 */
function CloseToTraySync() {
  useEffect(() => {
    if (!isTauri()) {
      return;
    }
    void syncCloseToTray(loadSettings().closeToTray).catch(() => {
      /* The default (enabled) stands if the sync cannot reach Rust. */
    });
  }, []);

  return null;
}

function RequireAuth({ children }: { children: ReactNode }) {
  const { status } = useAuth();
  const location = useLocation();

  if (status === "loading") {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner className="h-6 w-6 text-slate-400" />
      </div>
    );
  }

  if (status === "anonymous") {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }

  return <>{children}</>;
}

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <AuthProvider>
          <ToastProvider>
            <CloseToTraySync />
            <DeepLinkHandler />
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route
                element={
                  <RequireAuth>
                    <Layout />
                  </RequireAuth>
                }
              >
                <Route path="/" element={<DashboardPage />} />
                <Route path="/miners" element={<MinersPage />} />
                <Route path="/analytics" element={<AnalyticsPage />} />
                <Route path="/settings" element={<SettingsPage />} />
              </Route>
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </ToastProvider>
        </AuthProvider>
      </BrowserRouter>
    </ThemeProvider>
  );
}
