import type { ReactNode } from "react";
import { cn } from "../lib/utils";

export interface Column<T> {
  key: string;
  header: ReactNode;
  render: (row: T) => ReactNode;
  className?: string;
}

export function DataTable<T>({
  columns,
  rows,
  rowKey,
  empty,
  onRowClick,
}: {
  columns: Array<Column<T>>;
  rows: T[];
  rowKey: (row: T) => string;
  empty?: ReactNode;
  onRowClick?: (row: T) => void;
}) {
  if (rows.length === 0 && empty) {
    return <div className="px-5 py-10 text-center text-sm text-slate-500 dark:text-slate-400">{empty}</div>;
  }

  return (
    <div className="scrollbar-slim overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-slate-100 text-left dark:border-slate-800">
            {columns.map((column) => (
              <th
                key={column.key}
                className={cn(
                  "whitespace-nowrap px-5 py-3 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400",
                  column.className,
                )}
              >
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={rowKey(row)}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              className={cn(
                "border-b border-slate-50 last:border-0 dark:border-slate-800/60",
                onRowClick && "cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50",
              )}
            >
              {columns.map((column) => (
                <td key={column.key} className={cn("px-5 py-3 text-slate-700 dark:text-slate-300", column.className)}>
                  {column.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
