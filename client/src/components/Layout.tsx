import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  BarChart3,
  Bell,
  ChevronLeft,
  ChevronRight,
  LayoutDashboard,
  LogOut,
  Menu,
  Moon,
  Pickaxe,
  Search,
  Settings as SettingsIcon,
  Sun,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useAuth } from "../lib/auth";
import { cn } from "../lib/utils";
import { useTheme } from "./theme";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  end?: boolean;
}

const NAV_GROUPS: Array<{ caption: string; items: NavItem[] }> = [
  {
    caption: "Dashboard",
    items: [
      { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
      { to: "/analytics", label: "Analytics", icon: BarChart3 },
    ],
  },
  {
    caption: "Mining",
    items: [{ to: "/miners", label: "Miners", icon: Pickaxe }],
  },
  {
    caption: "Account",
    items: [{ to: "/settings", label: "Settings", icon: SettingsIcon }],
  },
];

const PAGE_META: Record<string, { title: string; trail: string[] }> = {
  "/": { title: "Dashboard", trail: ["Home", "Dashboard"] },
  "/analytics": { title: "Analytics", trail: ["Home", "Dashboard", "Analytics"] },
  "/miners": { title: "Miners", trail: ["Home", "Mining", "Miners"] },
  "/settings": { title: "Settings", trail: ["Home", "Account", "Settings"] },
};

function Topbar({
  collapsed,
  onToggle,
}: {
  collapsed: boolean;
  onToggle: () => void;
}) {
  const { user, signOut } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();

  async function handleSignOut() {
    await signOut();
    navigate("/login", { replace: true });
  }

  return (
    <header className="flex h-16 shrink-0 items-center bg-brand-500 text-white">
      {/* Brand segment sits over the sidebar column. */}
      <div className={cn("flex h-full shrink-0 items-center gap-2.5 px-4", collapsed ? "w-16" : "w-60")}>
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white/15">
          <Pickaxe className="h-4.5 w-4.5" />
        </div>
        {!collapsed && <span className="text-sm font-semibold tracking-wide">TokenMiner</span>}
      </div>

      <div className="flex h-full flex-1 items-center gap-3 px-4">
        <button
          type="button"
          onClick={onToggle}
          className="flex h-9 w-9 items-center justify-center rounded-lg text-white/90 hover:bg-white/15"
          aria-label="Toggle sidebar"
        >
          <Menu className="h-5 w-5" />
        </button>

        <div className="relative hidden max-w-sm flex-1 md:block">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            className="h-10 w-full rounded-lg bg-white pl-9 pr-16 text-sm text-slate-700 placeholder:text-slate-400 focus:outline-none"
            placeholder="Search"
            readOnly
          />
          <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 rounded border border-slate-200 px-1.5 py-0.5 text-[10px] text-slate-400">
            Ctrl + K
          </span>
        </div>

        <div className="ml-auto flex items-center gap-1.5">
          <button
            type="button"
            onClick={toggleTheme}
            className="flex h-9 w-9 items-center justify-center rounded-lg text-white/90 hover:bg-white/15"
            aria-label="Toggle theme"
          >
            {theme === "dark" ? <Sun className="h-4.5 w-4.5" /> : <Moon className="h-4.5 w-4.5" />}
          </button>

          <button
            type="button"
            className="relative flex h-9 w-9 items-center justify-center rounded-lg text-white/90 hover:bg-white/15"
            aria-label="Notifications"
          >
            <Bell className="h-4.5 w-4.5" />
          </button>

          <div className="ml-1 flex items-center gap-2 rounded-full bg-white/15 py-1 pl-1 pr-3">
            <div className="flex h-7 w-7 items-center justify-center rounded-full bg-white text-xs font-semibold text-brand-600">
              {(user?.displayName ?? user?.email ?? "T").charAt(0).toUpperCase()}
            </div>
            <span className="hidden max-w-32 truncate text-xs sm:block">
              {user?.displayName ?? user?.email ?? "Account"}
            </span>
            <button
              type="button"
              onClick={handleSignOut}
              className="text-white/80 hover:text-white"
              aria-label="Sign out"
              title="Sign out"
            >
              <LogOut className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}

function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  return (
    <aside
      className={cn(
        "flex shrink-0 flex-col border-r border-slate-200 bg-white transition-all duration-200 dark:border-slate-800 dark:bg-slate-900",
        collapsed ? "w-16" : "w-60",
      )}
    >
      <nav className="scrollbar-slim flex-1 overflow-y-auto px-3 py-4">
        {NAV_GROUPS.map((group) => (
          <div key={group.caption} className="mb-5">
            {!collapsed && (
              <p className="px-2 pb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                {group.caption}
              </p>
            )}
            <div className="space-y-1">
              {group.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  title={collapsed ? item.label : undefined}
                  className={({ isActive }) =>
                    cn(
                      "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                      isActive
                        ? "bg-brand-50 font-medium text-brand-500 dark:bg-brand-950 dark:text-brand-300"
                        : "text-slate-500 hover:bg-slate-50 hover:text-brand-500 dark:text-slate-400 dark:hover:bg-slate-800",
                      collapsed && "justify-center px-0",
                    )
                  }
                >
                  <item.icon className="h-4.5 w-4.5 shrink-0" />
                  {!collapsed && <span>{item.label}</span>}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      <button
        type="button"
        onClick={onToggle}
        className="flex items-center justify-center gap-2 border-t border-slate-200 px-3 py-3 text-xs text-slate-400 hover:text-brand-500 dark:border-slate-800"
      >
        {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        {!collapsed && <span>Collapse</span>}
      </button>
    </aside>
  );
}

/** The blue band under the topbar carrying the breadcrumb and page title. */
function PageBand({ trail, title }: { trail: string[]; title: string }) {
  return (
    <div className="bg-brand-500 px-6 pb-16 pt-4 text-white">
      <nav className="flex items-center gap-2 text-xs text-white/75">
        {trail.map((crumb, index) => (
          <span key={crumb} className="flex items-center gap-2">
            {index > 0 && <span className="text-white/40">›</span>}
            <span className={index === trail.length - 1 ? "text-white" : undefined}>{crumb}</span>
          </span>
        ))}
      </nav>
      <h1 className="mt-3 text-2xl font-semibold text-white">{title}</h1>
    </div>
  );
}

export function Layout() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  useEffect(() => {
    if (localStorage.getItem("tokenminer.sidebarCollapsed") === "true") {
      setCollapsed(true);
    }
  }, []);

  function toggle() {
    setCollapsed((current) => {
      localStorage.setItem("tokenminer.sidebarCollapsed", String(!current));
      return !current;
    });
  }

  const meta = PAGE_META[location.pathname] ?? { title: "Dashboard", trail: ["Home", "Dashboard"] };

  return (
    <div className="flex h-full flex-col">
      <Topbar collapsed={collapsed} onToggle={toggle} />
      <div className="flex min-h-0 flex-1">
        <Sidebar collapsed={collapsed} onToggle={toggle} />
        <div className="flex min-w-0 flex-1 flex-col bg-[#f4f6fa] dark:bg-slate-950">
          <PageBand trail={meta.trail} title={meta.title} />
          {/* Pulled up so the first row of cards overlaps the blue band, as in Able Pro. */}
          <main className="scrollbar-slim -mt-12 flex-1 overflow-y-auto px-6 pb-8">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  );
}
