import { useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom';
import { Icon } from './icons';
import type { IconName } from './icons';
import { useAuth } from '../lib/auth';
import { initials } from '../lib/format';

interface NavItem {
  to: string;
  label: string;
  icon: IconName;
}

interface NavGroup {
  title: string;
  items: NavItem[];
}

const NAVIGATION: NavGroup[] = [
  {
    title: 'Overview',
    items: [{ to: '/', label: 'Dashboard', icon: 'dashboard' }],
  },
  {
    title: 'Accounts',
    items: [{ to: '/users', label: 'Users', icon: 'users' }],
  },
  {
    title: 'Mining',
    items: [
      { to: '/resources/coins', label: 'Coins', icon: 'coins' },
      { to: '/resources/pools', label: 'Pools', icon: 'pool' },
      { to: '/resources/mining-algos', label: 'Algorithms', icon: 'cpu' },
    ],
  },
  {
    title: 'Treasury',
    items: [
      { to: '/resources/conversion-providers', label: 'Conversion providers', icon: 'swap' },
      { to: '/resources/conversion-routes', label: 'Conversion routes', icon: 'route' },
      { to: '/conversions', label: 'Conversions', icon: 'layers' },
      { to: '/pool-payouts', label: 'Pool payouts', icon: 'wallet' },
    ],
  },
  {
    title: 'Providers',
    items: [
      { to: '/resources/llm-providers', label: 'LLM providers', icon: 'shield' },
      { to: '/provider-deposits', label: 'Provider deposits', icon: 'package' },
    ],
  },
  {
    title: 'System',
    items: [{ to: '/settings', label: 'Configuration', icon: 'settings' }],
  },
];

export function Layout() {
  const { user, signOut } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);

  // The shell's layout is driven by body classes so the CSS owns the transitions.
  useEffect(() => {
    document.body.classList.toggle('nav-collapsed', collapsed);
  }, [collapsed]);

  useEffect(() => {
    document.body.classList.toggle('nav-open', mobileOpen);
  }, [mobileOpen]);

  // A route change always closes the mobile drawer.
  useEffect(() => {
    setMobileOpen(false);
    setProfileOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    return () => {
      document.body.classList.remove('nav-collapsed', 'nav-open');
    };
  }, []);

  async function handleSignOut() {
    await signOut();
    navigate('/login', { replace: true });
  }

  return (
    <>
      <div className="sidebar-backdrop" onClick={() => setMobileOpen(false)} aria-hidden="true" />

      <aside className="dlabnav">
        <div className="dlabnav-scroll">
          <NavLink to="/" className="brand-logo">
            <span className="brand-mark">T</span>
            <span className="brand-text">
              <strong>TokenMiner</strong>
              <small>Management portal</small>
            </span>
          </NavLink>

          <div className="sidebar-profile">
            <span className="avatar avatar-sm">{initials(user?.displayName ?? user?.email)}</span>
            <div className="brand-text">
              <strong style={{ fontSize: '0.8125rem' }}>{user?.displayName ?? 'Operator'}</strong>
              <small>{user?.roles.join(', ') ?? ''}</small>
            </div>
          </div>

          {NAVIGATION.map((group) => (
            <div key={group.title}>
              <div className="menu-title">{group.title}</div>
              <ul>
                {group.items.map((item) => (
                  <li className="nav-item" key={item.to}>
                    <NavLink
                      to={item.to}
                      end={item.to === '/'}
                      className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
                      title={item.label}
                    >
                      <span className="nav-icon">
                        <Icon name={item.icon} />
                      </span>
                      <span className="nav-text">{item.label}</span>
                    </NavLink>
                  </li>
                ))}
              </ul>
            </div>
          ))}

          <div className="sidebar-footer">
            <p className="mb-0">
              <strong>TokenMiner</strong> operator console
            </p>
            <p className="mb-0 fs-12">Mining &rarr; treasury &rarr; provider credit</p>
          </div>
        </div>
      </aside>

      <header className="nav-header">
        <div className="nav-control">
          <button
            type="button"
            className="hamburger"
            aria-label="Toggle navigation"
            onClick={() => {
              if (window.matchMedia('(max-width: 991.98px)').matches) {
                setMobileOpen((open) => !open);
              } else {
                setCollapsed((value) => !value);
              }
            }}
            style={{ border: 0, background: 'transparent', padding: 0 }}
          >
            <span className="line" />
            <span className="line" />
            <span className="line" />
          </button>
        </div>

        <div className="header-actions">
          <div style={{ position: 'relative' }}>
            <button
              type="button"
              className="header-profile"
              onClick={() => setProfileOpen((open) => !open)}
              aria-expanded={profileOpen}
            >
              <span className="avatar avatar-sm">{initials(user?.displayName ?? user?.email)}</span>
              <span className="profile-meta">
                <strong>{user?.displayName ?? 'Operator'}</strong>
                <small>{user?.email}</small>
              </span>
              <Icon name="chevronDown" />
            </button>

            {profileOpen && (
              <div
                className="card"
                style={{
                  position: 'absolute',
                  right: 0,
                  top: 'calc(100% + 0.5rem)',
                  minWidth: '15rem',
                  marginBottom: 0,
                  padding: '0.5rem',
                  zIndex: 200,
                }}
              >
                <div style={{ padding: '0.5rem 0.75rem 0.75rem' }}>
                  <div className="cell-strong">{user?.displayName ?? 'Operator'}</div>
                  <div className="field-hint mb-0">{user?.email}</div>
                  <div className="field-hint mb-0">Roles: {user?.roles.join(', ') || 'none'}</div>
                </div>
                <button type="button" className="btn btn-outline-secondary w-100" onClick={handleSignOut}>
                  <Icon name="logout" />
                  Sign out
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="content-body">
        <div className="container-fluid">
          <Outlet />
        </div>
      </main>
    </>
  );
}
