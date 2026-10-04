import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { CheckCircle2, Info, X, XCircle } from "lucide-react";
import { cn } from "../lib/utils";

type ToastTone = "success" | "error" | "info";

interface Toast {
  id: number;
  tone: ToastTone;
  message: string;
}

interface ToastContextValue {
  push: (tone: ToastTone, message: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

const TONE_STYLES: Record<ToastTone, { ring: string; icon: typeof Info }> = {
  success: { ring: "border-emerald-200 dark:border-emerald-900", icon: CheckCircle2 },
  error: { ring: "border-red-200 dark:border-red-900", icon: XCircle },
  info: { ring: "border-slate-200 dark:border-slate-700", icon: Info },
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const remove = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback(
    (tone: ToastTone, message: string) => {
      const id = Date.now() + Math.random();
      setToasts((current) => [...current, { id, tone, message }]);
      window.setTimeout(() => remove(id), 5000);
    },
    [remove],
  );

  const value = useMemo<ToastContextValue>(() => ({ push }), [push]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-80 flex-col gap-2">
        {toasts.map((toast) => {
          const { ring, icon: Icon } = TONE_STYLES[toast.tone];
          return (
            <div
              key={toast.id}
              className={cn(
                "pointer-events-auto flex items-start gap-3 rounded-lg border bg-white p-3 shadow-lg dark:bg-slate-900",
                ring,
              )}
            >
              <Icon className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" />
              <p className="flex-1 text-sm text-slate-700 dark:text-slate-200">{toast.message}</p>
              <button
                type="button"
                onClick={() => remove(toast.id)}
                className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                aria-label="Dismiss"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used inside a ToastProvider.");
  }
  return context;
}
