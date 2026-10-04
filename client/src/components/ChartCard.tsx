import type { ReactNode } from "react";
import Chart from "react-apexcharts";
import type { ApexOptions } from "apexcharts";
import { Card, CardHeader } from "./ui";

const BRAND = "#6366f1";

export type ChartType = "line" | "area" | "bar" | "donut" | "pie" | "radialBar" | "scatter" | "heatmap";

export function ChartCard({
  title,
  subtitle,
  action,
  type,
  series,
  options,
  height = 300,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  type: ChartType;
  series: ApexOptions["series"];
  options?: ApexOptions;
  height?: number;
}) {
  const merged: ApexOptions = {
    chart: {
      toolbar: { show: false },
      fontFamily: "inherit",
      foreColor: "#94a3b8",
      type,
    },
    colors: [BRAND, "#0ea5e9", "#16a34a", "#f59e0b"],
    dataLabels: { enabled: false },
    stroke: { curve: "smooth", width: 2 },
    grid: { borderColor: "rgba(148,163,184,0.2)", strokeDashArray: 4 },
    xaxis: { axisBorder: { show: false }, axisTicks: { show: false } },
    legend: { position: "bottom", markers: { size: 6 } },
    tooltip: { theme: "light" },
    ...options,
  };

  return (
    <Card>
      {title || subtitle || action ? <CardHeader title={title} subtitle={subtitle} action={action} /> : null}
      <div className="px-3 py-4">
        <Chart type={type} series={series} options={merged} height={height} />
      </div>
    </Card>
  );
}
