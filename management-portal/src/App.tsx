import type { ReactNode } from 'react';
import { BrowserRouter, Link, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from './lib/auth';
import { EmptyState, ToastProvider } from './components/ui';
import { Layout } from './components/Layout';
import { LoginPage } from './pages/LoginPage';
import { DashboardPage } from './pages/DashboardPage';
import { ResourcePage } from './pages/ResourcePage';
import { ConversionsPage, PoolPayoutsPage, ProviderDepositsPage } from './pages/LedgerPages';
import { SettingsPage } from './pages/SettingsPage';

/** The portal is operator-only, so a session without the admin role never reaches a screen. */
function RequireAdmin({ children }: { children: ReactNode }) {
  const { status, isAdmin } = useAuth();
  const location = useLocation();

  if (status === 'loading') {
    return (
      <div className="login-page">
        <div className="spinner lg" />
      </div>
    );
  }

  if (status === 'anonymous') {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }

  if (!isAdmin) {
    // The sign-in screen explains the missing role rather than looping.
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

function NotFound() {
  return (
    <div className="login-page">
      <div className="login-card">
        <EmptyState
          icon="alert"
          title="Page not found"
          message="That route is not part of the portal."
          action={
            <Link to="/" className="btn btn-primary btn-sm">
              Back to the dashboard
            </Link>
          }
        />
      </div>
    </div>
  );
}

export function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <ToastProvider>
          <Routes>
            <Route path="/login" element={<LoginPage />} />

            <Route
              element={
                <RequireAdmin>
                  <Layout />
                </RequireAdmin>
              }
            >
              <Route path="/" element={<DashboardPage />} />
              <Route path="/resources/:resourceKey" element={<ResourcePage />} />
              <Route path="/conversions" element={<ConversionsPage />} />
              <Route path="/pool-payouts" element={<PoolPayoutsPage />} />
              <Route path="/provider-deposits" element={<ProviderDepositsPage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Route>

            <Route path="*" element={<NotFound />} />
          </Routes>
        </ToastProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}
