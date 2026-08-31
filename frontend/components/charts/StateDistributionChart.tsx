"use client";

import ReactECharts from "echarts-for-react";
import { useEffect, useRef, useState } from "react";
import { useChartColors } from "@/lib/chartTheme";

interface Props {
  data: Record<string, number>;
  /** Called on each tick while live-refresh is on — pass the page's own reload fn. */
  onRefresh?: () => void;
}

type Segment = "all" | "active" | "terminal";
const ACTIVE = new Set(["Current", "30DPD", "60DPD", "90DPD", "Default"]);
const TERMINAL = new Set(["Prepaid", "Closed"]);

const STATE_TOKEN: Record<string, string> = {
  Current: "--color-risk-low",
  "30DPD": "--color-risk-medium",
  "60DPD": "--chart-3",
  "90DPD": "--color-risk-high",
  Default: "--chart-4",
  Prepaid: "--color-survival",
  Closed: "--chart-5",
};

// Bars animate to their new height on every data change (default ECharts
// update animation) and, when "Live" is toggled on, the chart re-polls the
// portfolio on an interval the same way the ECharts "dynamic-data" example
// pushes fresh points on a timer:
// https://echarts.apache.org/examples/en/editor.html?c=dynamic-data
export function StateDistributionChart({ data, onRefresh }: Props) {
  const c = useChartColors();
  const [segment, setSegment] = useState<Segment>("all");
  const [live, setLive] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (live && onRefresh) {
      timer.current = setInterval(onRefresh, 8000);
    }
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [live, onRefresh]);

  const entries = Object.entries(data)
    .filter(([k]) => segment === "all" || (segment === "active" ? ACTIVE.has(k) : TERMINAL.has(k)))
    .sort((a, b) => b[1] - a[1]);

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      backgroundColor: c["--chart-tooltip-bg"],
      borderColor: "transparent",
      textStyle: { color: c["--chart-tooltip-text"], fontFamily: "var(--font-geist-sans)", fontSize: 13 },
      formatter: (p: { name: string; value: number }[]) =>
        `<b>${p[0].name}</b><br/>${p[0].value.toLocaleString()} loans`,
    },
    grid: { left: 12, right: 16, top: 8, bottom: 24, containLabel: true },
    xAxis: {
      type: "category",
      data: entries.map(([k]) => k),
      axisLabel: { color: c["--color-text-tertiary"], fontSize: 11, fontFamily: "var(--font-geist-mono)" },
      axisLine: { lineStyle: { color: c["--color-border"] } },
      axisTick: { show: false },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: c["--color-text-tertiary"], fontSize: 10, fontFamily: "var(--font-geist-mono)" },
      splitLine: { lineStyle: { color: c["--chart-grid"] } },
    },
    series: [
      {
        type: "bar",
        barMaxWidth: 40,
        borderRadius: [4, 4, 0, 0],
        data: entries.map(([k, v]) => ({
          value: v,
          itemStyle: { color: c[STATE_TOKEN[k] as keyof typeof c] ?? c["--color-accent"] },
        })),
        label: { show: false },
        animationDurationUpdate: 500,
        animationEasingUpdate: "cubicOut",
      },
    ],
  };

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "flex-end", gap: 6, marginBottom: 6 }}>
        {(["all", "active", "terminal"] as const).map(s => (
          <button
            key={s}
            onClick={() => setSegment(s)}
            style={{
              fontSize: 11, padding: "3px 9px", borderRadius: 5, cursor: "pointer",
              border: `1px solid ${segment === s ? "var(--color-accent)" : "var(--color-border)"}`,
              background: segment === s ? "var(--color-accent-muted)" : "transparent",
              color: segment === s ? "var(--color-accent)" : "var(--color-text-tertiary)",
              textTransform: "capitalize", transition: "all 150ms ease",
            }}
          >{s}</button>
        ))}
        {onRefresh && (
          <button
            onClick={() => setLive(l => !l)}
            title="Poll the portfolio every 8s"
            style={{
              display: "flex", alignItems: "center", gap: 5, fontSize: 11, padding: "3px 9px", borderRadius: 5,
              cursor: "pointer", border: `1px solid ${live ? "var(--color-risk-low)" : "var(--color-border)"}`,
              background: live ? "rgba(34,197,94,.08)" : "transparent",
              color: live ? "var(--color-risk-low)" : "var(--color-text-tertiary)", transition: "all 150ms ease",
            }}
          >
            {live && <span className="gs-live-dot" />}
            Live
          </button>
        )}
      </div>
      <ReactECharts option={option} style={{ height: 210 }} notMerge={false} lazyUpdate />
    </div>
  );
}
