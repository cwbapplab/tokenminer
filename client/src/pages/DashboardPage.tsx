import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { ArrowDownLeft, ArrowUpRight, Cpu, Gauge, Minus, Pickaxe, Zap } from "lucide-react";
import type { ApexOptions } from "apexcharts";
import Chart from "react-apexcharts";
import { Card, EmptyState, Spinner } from "../components/ui";
import { MiniChart } from "../components/MiniChart";
import { useMiners } from "../hooks/useMiners";
import { getAnalytics } from "../lib/session";
import { cn, formatHashrate, formatPearlRate, formatUsd } from "../lib/utils";
import type { MiningAnalytics } from "../lib/types";

const SAMPLE_INTERVAL_MS = 2000;
const MAX_SAMPLES = 30;

type Tab = "overview" | "quantus" | "pearl";

export function DashboardPage() {
  const miners = useMiners();
  const [analytics, setAnalytics] = useState<MiningAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [samples, setSamples] = useState<Array<{ t: number; quantus: number; pearl: number }>>([]);
  const [tab, setTab] = useState<Tab>("overview");
  const minersRef = useRef(miners);
  minersRef.current = miners;

  useEffect(() => {
    let cancelled = false;
    void getAnalytics()
      .then((data) => {
        if (!cancelled) setAnalytics(data);
      })
      .catch(() => {
        /* Not signed in or API unavailable — cards fall back to zeros. */
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setSamples((current) =>
        [
          ...current,
          { t: Date.now(), quantus: minersRef.current.quantus.hashrate, pearl: minersRef.current.pearl.hashrate },
        ].slice(-MAX_SAMPLES),
      );
    }, SAMPLE_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, []);

  // Not the sum of the two engines, and deliberately. A Quantus hash and a Pearl tile are different
  // quantities — the Pearl one is 16 x 16 x 4096 int8 MACs — so adding them produces a number with
  // no unit behind it, and a "Total Hashrate" that jumps when you start the GPU engine rather than
  // when you do more work. Each engine is reported in its own unit and the combined figure is
  // labelled as being two engines rather than one rate.
  const activeMiners = [miners.quantus, miners.pearl].filter((m) => m.state === "running").length;

  const quantusSeries = samples.map((s) => s.quantus);
  const pearlSeries = samples.map((s) => s.pearl);

  const hashrateDelta = useMemo(() => {
    if (samples.length < 6) return null;
    const past = samples[0].quantus;
    const now = samples[samples.length - 1].quantus;
    if (past <= 0) return null;
    return ((now - past) / past) * 100;
  }, [samples]);

  const totals = analytics?.totals ?? { last30Usd: 0, last60Usd: 0, allTimeUsd: 0 };
  const previous30 = Math.max(0, totals.last60Usd - totals.last30Usd);
  const earningsDelta = previous30 > 0 ? ((totals.last30Usd - previous30) / previous30) * 100 : null;

  // Which engines are running, not how the work divides between them: the two rates have no common
  // unit to take a ratio of. Each engine's own rate is printed underneath in its own unit.
  const share = activeMiners > 0 && miners.quantus.state === "running" ? 100 / activeMiners : 0;

  // Each engine's tab plots its own rate on its own axis, in its own unit.
//
// The overview plots both, which cannot be on one scale: a Pearl rate is tens of millions of tiles
// per second and a Quantus rate is orders of magnitude smaller, so a shared axis draws Quantus as a
// flat line on the floor — a chart that looks like a stopped engine. Normalising each against its
// own peak over the window shows both engines' shape (running, stalling, restarting) without
// implying that the two numbers are comparable, and the exact rates are on the cards above.
  const peak = (series: number[]) => series.reduce((max, v) => (v > max ? v : max), 0);
  const relative = (series: number[]) => {
    const top = peak(series);
    return top > 0 ? series.map((v) => (v / top) * 100) : series.map(() => 0);
  };

  const overviewSeries = [
    { name: "Quantus", data: relative(quantusSeries) },
    { name: "Pearl", data: relative(pearlSeries) },
  ];

  const chartSeries = {
    overview: overviewSeries,
    quantus: [{ name: "Quantus", data: quantusSeries }],
    pearl: [{ name: "Pearl", data: pearlSeries }],
  }[tab];
  const chartIsRelative = tab === "overview";
  const formatChartValue = chartIsRelative
    ? (value: number) => `${Math.round(value)}%`
    : tab === "quantus"
      ? formatHashrate
      : formatPearlRate;

  // The y-axis and tooltip formatter follows the tab: a Pearl axis labelled in H/s is the same
  // category error as summing the two engines, just drawn instead of printed.
  const chartOptions: ApexOptions = {
    chart: { toolbar: { show: false }, fontFamily: "inherit", foreColor: "#94a3b8" },
    colors: ["#4680ff"],
    plotOptions: { bar: { columnWidth: "50%", borderRadius: 3 } },
    dataLabels: { enabled: false },
    grid: { borderColor: "rgba(148,163,184,0.18)", strokeDashArray: 4 },
    xaxis: { labels: { show: false }, axisBorder: { show: false }, axisTicks: { show: false } },
    yaxis: { labels: { formatter: formatChartValue } },
    tooltip: { y: { formatter: formatChartValue } },
  };

  return (
    <div className="grid grid-cols-12 gap-4">
      <KpiCard
        title="Both engines"
        value={`${formatHashrate(miners.quantus.hashrate)} · ${formatPearlRate(miners.pearl.hashrate)}`}
        delta={null}
        hint={`${activeMiners} of 2 engines online`}
        chart={<MiniChart kind="bar" data={quantusSeries} />}
      />
      <KpiCard
        title="Quantus"
        value={formatHashrate(miners.quantus.hashrate)}
        delta={hashrateDelta}
        hint={miners.quantus.state}
        accent="success"
        chart={<MiniChart kind="line" data={quantusSeries} />}
      />
      <KpiCard
        title="Pearl"
        value={formatPearlRate(miners.pearl.hashrate)}
        delta={null}
        hint={miners.pearl.state}
        accent="warning"
        chart={<MiniChart kind="dots" data={pearlSeries} />}
      />

      {/* Fourth column: dark stat card + blue balance card, as in Able Pro. */}
      <div className="col-span-12 flex flex-col gap-4 md:col-span-6 xl:col-span-3">
        <div className="rounded-lg bg-gradient-to-br from-slate-800 to-slate-900 p-5 text-white shadow-sm">
          <div className="flex items-start justify-between">
            <h5 className="text-sm font-semibold text-white">Engine split</h5>
            <span className="text-sm font-semibold text-white">{Math.round(share)}%</span>
          </div>
          <div className="my-4">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-white/10">
              <Zap className="h-4 w-4" />
            </div>
          </div>
          <p className="text-xs text-white/80">
            Quantus {formatHashrate(miners.quantus.hashrate)} · Pearl {formatPearlRate(miners.pearl.hashrate)}
          </p>
          <div className="mt-2 flex h-1.5 overflow-hidden rounded-full bg-white/10">
            <div className="bg-brand-400" style={{ width: `${share}%` }} />
            <div className="bg-emerald-400" style={{ width: `${100 - share}%` }} />
          </div>
        </div>

        <div className="flex items-center justify-between rounded-lg bg-brand-500 p-5 text-white shadow-sm">
          <div>
            <p className="text-xs text-white/80">Earnings · all time</p>
            <p className="mt-1 text-xl font-semibold text-white">{formatUsd(totals.allTimeUsd)}</p>
          </div>
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-white/15">
            <Gauge className="h-4 w-4" />
          </div>
        </div>
      </div>

      {/* Tabbed chart + stats list */}
      <div className="col-span-12">
        <Card>
          <div className="border-b border-slate-100 px-5 pt-4 dark:border-slate-800">
            <div className="flex gap-6">
              {(
                [
                  ["overview", "Both (relative)"],
                  ["quantus", "Quantus"],
                  ["pearl", "Pearl"],
                ] as Array<[Tab, string]>
              ).map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setTab(key)}
                  className={cn(
                    "-mb-px border-b-2 pb-3 text-sm transition-colors",
                    tab === key
                      ? "border-brand-500 font-medium text-brand-500"
                      : "border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400",
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-12 gap-4 p-5">
            <div className="col-span-12 lg:col-span-8">
              <div className="mb-3 flex items-center gap-2">
                <span className="rounded-md bg-brand-500 px-3 py-1 text-xs font-medium text-white">Live</span>
                {chartIsRelative ? (
                  <span className="rounded-md border border-slate-200 px-3 py-1 text-xs text-slate-500 dark:border-slate-700">
                    % of each engine&apos;s own peak — the two rates have no common unit
                  </span>
                ) : (
                  <span className="rounded-md border border-slate-200 px-3 py-1 text-xs text-slate-500 dark:border-slate-700">
                    Last 60s
                  </span>
                )}
              </div>
              <Chart type="bar" height={280} series={chartSeries} options={chartOptions} />
            </div>

            <div className="col-span-12 lg:col-span-4">
              <ul className="divide-y divide-slate-100 dark:divide-slate-800">
                <StatRow icon={Cpu} label="CPU hashrate" value={formatHashrate(miners.quantus.cpuHashrate)} />
                <StatRow icon={Zap} label="Quantus GPU hashrate" value={formatHashrate(miners.quantus.gpuHashrate)} />
                <StatRow icon={Pickaxe} label="Pearl tile rate" value={formatPearlRate(miners.pearl.gpuHashrate)} />
                <StatRow icon={Pickaxe} label="Workers" value={String(miners.quantus.workers + miners.pearl.workers)} />
                <StatRow icon={Gauge} label="Active jobs" value={String(miners.quantus.activeJobs)} />
              </ul>
            </div>
          </div>
        </Card>
      </div>

      {/* Devices / earnings table */}
      <div className="col-span-12">
        <Card>
          <div className="flex items-center justify-between px-5 py-4">
            <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">Device earnings</h3>
            {earningsDelta !== null && <Delta value={earningsDelta} />}
          </div>
          {loading ? (
            <div className="flex justify-center py-10">
              <Spinner className="h-6 w-6 text-slate-400" />
            </div>
          ) : (analytics?.hardware.length ?? 0) === 0 ? (
            <EmptyState icon={Cpu} title="No devices yet" message="Start mining to register this device." />
          ) : (
            <ul className="divide-y divide-slate-100 dark:divide-slate-800">
              {analytics?.hardware.map((device) => (
                <li key={device.userHardwareId} className="flex items-center justify-between px-5 py-3">
                  <span className="font-mono text-xs text-slate-500">{device.userHardwareId}</span>
                  <span className="flex gap-6 text-sm">
                    <span className="text-slate-400">30d {formatUsd(device.last30Usd)}</span>
                    <span className="font-medium text-slate-700 dark:text-slate-200">
                      all {formatUsd(device.allTimeUsd)}
                    </span>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  );
}

function KpiCard({
  title,
  value,
  delta,
  hint,
  chart,
  accent = "brand",
}: {
  title: string;
  value: string;
  delta: number | null;
  hint?: string;
  chart: ReactNode;
  accent?: "brand" | "success" | "warning";
}) {
  return (
    <div className="col-span-12 md:col-span-6 xl:col-span-3">
      <Card className="flex h-full flex-col p-5">
        <div className="flex items-center justify-between">
          <h5 className="text-base font-semibold text-slate-800 dark:text-slate-100">{title}</h5>
          <span className="rounded-md border border-slate-200 px-2 py-1 text-xs capitalize text-slate-500 dark:border-slate-700 dark:text-slate-400">
            {hint ?? "Live"}
          </span>
        </div>
        <div className="my-3 flex-1">{chart}</div>
        <div className="text-center">
          <p className="text-lg font-semibold text-slate-800 dark:text-slate-100">
            {value}
            {delta !== null && <Delta value={delta} className="ml-2 align-middle" />}
          </p>
        </div>
        <button
          type="button"
          className={cn(
            "mt-3 w-full rounded-md border py-1.5 text-xs font-medium",
            accent === "brand" && "border-brand-200 text-brand-500 hover:bg-brand-50 dark:border-brand-900",
            accent === "success" && "border-emerald-200 text-emerald-600 hover:bg-emerald-50 dark:border-emerald-900",
            accent === "warning" && "border-amber-200 text-amber-600 hover:bg-amber-50 dark:border-amber-900",
          )}
        >
          View More
        </button>
      </Card>
    </div>
  );
}

function Delta({ value, className }: { value: number; className?: string }) {
  const positive = value >= 0;
  const Icon = value === 0 ? Minus : positive ? ArrowUpRight : ArrowDownLeft;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-0.5 text-xs font-medium",
        value === 0 ? "text-slate-400" : positive ? "text-emerald-600" : "text-red-500",
        className,
      )}
    >
      <Icon className="h-3.5 w-3.5" />
      {Math.abs(value).toFixed(1)}%
    </span>
  );
}

function StatRow({ icon: Icon, label, value }: { icon: typeof Cpu; label: string; value: string }) {
  return (
    <li className="flex items-center gap-3 py-3">
      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-slate-100 text-slate-500 dark:bg-slate-800">
        <Icon className="h-4 w-4" />
      </span>
      <span className="flex-1 text-sm text-slate-500 dark:text-slate-400">{label}</span>
      <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">{value}</span>
    </li>
  );
}
