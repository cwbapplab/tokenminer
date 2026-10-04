import { useEffect, useState } from "react";
import { CalendarClock, CalendarDays, TrendingUp, Wallet } from "lucide-react";
import { Card, CardHeader, EmptyState, Spinner, StatCard } from "../components/ui";
import { DataTable, type Column } from "../components/DataTable";
import { getAnalytics } from "../lib/session";
import { formatUsd } from "../lib/utils";
import type { HardwareAnalytics, MiningAnalytics } from "../lib/types";

export function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<MiningAnalytics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    void getAnalytics()
      .then((data) => {
        if (!cancelled) {
          setAnalytics(data);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const columns: Array<Column<HardwareAnalytics>> = [
    {
      key: "device",
      header: "Device",
      render: (row) => <span className="font-mono text-xs">{row.userHardwareId}</span>,
    },
    { key: "last30", header: "Last 30 days", render: (row) => formatUsd(row.last30Usd) },
    { key: "last60", header: "Last 60 days", render: (row) => formatUsd(row.last60Usd) },
    { key: "all", header: "All time", render: (row) => formatUsd(row.allTimeUsd) },
  ];

  const totals = analytics?.totals ?? { last30Usd: 0, last60Usd: 0, allTimeUsd: 0 };

  return (
    <>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard label="Last 30 days" value={formatUsd(totals.last30Usd)} icon={CalendarDays} tone="brand" />
        <StatCard label="Last 60 days" value={formatUsd(totals.last60Usd)} icon={CalendarClock} tone="info" />
        <StatCard label="All time" value={formatUsd(totals.allTimeUsd)} icon={Wallet} tone="success" />
      </div>

      <Card className="mt-4">
        <CardHeader title="Per device" subtitle="Earnings broken down by hardware id" />
        {loading ? (
          <div className="flex justify-center py-10">
            <Spinner className="h-6 w-6 text-slate-400" />
          </div>
        ) : (
          <DataTable
            columns={columns}
            rows={analytics?.hardware ?? []}
            rowKey={(row) => row.userHardwareId}
            empty={
              <EmptyState
                icon={TrendingUp}
                title="No earnings yet"
                message="Once the pool accepts shares, earnings will show here."
              />
            }
          />
        )}
      </Card>
    </>
  );
}
