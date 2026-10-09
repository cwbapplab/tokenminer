import { useEffect, useRef } from "react";
import { cn } from "../lib/utils";

export interface LogLine {
  id: number;
  level: string;
  message: string;
  timestamp?: string;
}

const LEVEL_COLORS: Record<string, string> = {
  error: "text-red-400",
  warn: "text-amber-400",
  info: "text-slate-300",
  debug: "text-slate-500",
  trace: "text-slate-600",
};

export function LogConsole({ lines, className }: { lines: LogLine[]; className?: string }) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines.length]);

  return (
    <div
      ref={scrollRef}
      className={cn(
        "scrollbar-slim h-64 overflow-y-auto rounded-lg bg-slate-950 p-3 font-mono text-xs leading-relaxed",
        className,
      )}
    >
      {lines.length === 0 ? (
        <p className="text-slate-600">No output yet.</p>
      ) : (
        lines.map((line) => (
          <div key={line.id} className="flex gap-2 whitespace-pre-wrap break-all">
            {line.timestamp ? <span className="shrink-0 text-slate-600">{line.timestamp}</span> : null}
            <span className={cn("shrink-0 uppercase", LEVEL_COLORS[line.level] ?? "text-slate-300")}>
              {line.level}
            </span>
            <span className="text-slate-300">{line.message}</span>
          </div>
        ))
      )}
    </div>
  );
}
