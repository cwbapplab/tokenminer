import Chart from "react-apexcharts";
import type { ApexOptions } from "apexcharts";

export type MiniChartKind = "bar" | "line" | "dots" | "area";

const COLORS: Record<MiniChartKind, string> = {
  bar: "#4680ff",
  line: "#2ca87f",
  dots: "#e58a00",
  area: "#4680ff",
};

/** Tiny sparkline used inside the KPI cards — mirrors Able Pro's mini widgets. */
export function MiniChart({ kind, data, height = 70 }: { kind: MiniChartKind; data: number[]; height?: number }) {
  const color = COLORS[kind];

  const options: ApexOptions = {
    chart: {
      type: kind === "dots" ? "scatter" : kind === "bar" ? "bar" : "area",
      sparkline: { enabled: true },
      toolbar: { show: false },
      animations: { enabled: false },
    },
    colors: [color],
    plotOptions: kind === "bar" ? { bar: { columnWidth: "55%", borderRadius: 3 } } : {},
    stroke: { curve: "smooth", width: kind === "bar" || kind === "dots" ? 0 : 2.5 },
    fill:
      kind === "area"
        ? { type: "gradient", gradient: { opacityFrom: 0.35, opacityTo: 0.05 } }
        : { type: "solid" },
    markers: kind === "dots" ? { size: 6, strokeWidth: 0 } : { size: 0 },
    tooltip: { enabled: false },
  };

  const series =
    kind === "dots"
      ? [{ name: "value", data: data.map((y, x) => [x, y]) }]
      : [{ name: "value", data }];

  return <Chart type={kind === "dots" ? "scatter" : kind === "bar" ? "bar" : "area"} series={series} options={options} height={height} />;
}
