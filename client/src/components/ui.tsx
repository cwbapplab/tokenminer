import type { ButtonHTMLAttributes, ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { cn } from "../lib/utils";

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("pc-card", className)}>{children}</div>;
}

export function CardHeader({
  title,
  subtitle,
  action,
  className,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-start justify-between gap-4 border-b border-slate-100 px-5 py-4 dark:border-slate-800", className)}>
      <div>
        <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{title}</h3>
        {subtitle ? <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{subtitle}</p> : null}
      </div>
      {action}
    </div>
  );
}

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "success";
type ButtonSize = "sm" | "md";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-brand-600 text-white hover:bg-brand-700 focus-visible:outline-brand-600",
  secondary:
    "bg-white text-slate-700 border border-slate-300 hover:bg-slate-50 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-700 dark:hover:bg-slate-700",
  ghost: "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800",
  danger: "bg-red-600 text-white hover:bg-red-700 focus-visible:outline-red-600",
  success: "bg-emerald-600 text-white hover:bg-emerald-700 focus-visible:outline-emerald-600",
};

const BUTTON_SIZES: Record<ButtonSize, string> = {
  sm: "h-8 px-3 text-xs",
  md: "h-10 px-4 text-sm",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: LucideIcon;
  loading?: boolean;
}

export function Button({
  variant = "primary",
  size = "md",
  icon: Icon,
  loading = false,
  className,
  children,
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        className,
      )}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? <Spinner className="h-4 w-4" /> : Icon ? <Icon className="h-4 w-4" /> : null}
      {children}
    </button>
  );
}

type BadgeTone = "neutral" | "brand" | "success" | "warning" | "danger";

const BADGE_TONES: Record<BadgeTone, string> = {
  neutral: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300",
  brand: "bg-brand-50 text-brand-700 dark:bg-brand-950 dark:text-brand-300",
  success: "bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  warning: "bg-amber-50 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
  danger: "bg-red-50 text-red-700 dark:bg-red-950 dark:text-red-300",
};

export function Badge({ tone = "neutral", children }: { tone?: BadgeTone; children: ReactNode }) {
  return (
    <span className={cn("inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium", BADGE_TONES[tone])}>
      {children}
    </span>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg className={cn("animate-spin text-current", className)} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 0 1 8-8v4a4 4 0 0 0-4 4H4z" />
    </svg>
  );
}

export function EmptyState({
  icon: Icon,
  title,
  message,
  action,
}: {
  icon?: LucideIcon;
  title: string;
  message?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-12 text-center">
      {Icon ? (
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-100 text-slate-400 dark:bg-slate-800">
          <Icon className="h-6 w-6" />
        </div>
      ) : null}
      <div>
        <p className="text-sm font-semibold text-slate-800 dark:text-slate-100">{title}</p>
        {message ? <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{message}</p> : null}
      </div>
      {action}
    </div>
  );
}

export function PageHeader({ title, subtitle, action }: { title: string; subtitle?: string; action?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-50">{title}</h1>
        {subtitle ? <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">{subtitle}</p> : null}
      </div>
      {action}
    </div>
  );
}

type StatTone = "brand" | "success" | "warning" | "info";

const STAT_TONES: Record<StatTone, string> = {
  brand: "bg-brand-50 text-brand-600 dark:bg-brand-950 dark:text-brand-300",
  success: "bg-emerald-50 text-emerald-600 dark:bg-emerald-950 dark:text-emerald-300",
  warning: "bg-amber-50 text-amber-600 dark:bg-amber-950 dark:text-amber-300",
  info: "bg-sky-50 text-sky-600 dark:bg-sky-950 dark:text-sky-300",
};

export function StatCard({
  label,
  value,
  hint,
  icon: Icon,
  tone = "brand",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  icon: LucideIcon;
  tone?: StatTone;
}) {
  return (
    <Card className="p-5">
      <div className="flex items-center gap-4">
        <div className={cn("flex h-12 w-12 shrink-0 items-center justify-center rounded-xl", STAT_TONES[tone])}>
          <Icon className="h-6 w-6" />
        </div>
        <div className="min-w-0">
          <p className="truncate text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">{label}</p>
          <p className="mt-1 truncate text-2xl font-semibold text-slate-900 dark:text-slate-50">{value}</p>
          {hint ? <p className="mt-0.5 truncate text-xs text-slate-400">{hint}</p> : null}
        </div>
      </div>
    </Card>
  );
}
