"use client";

import ReactECharts from "echarts-for-react";
import { useState } from "react";
import { useChartColors, withAlpha } from "@/lib/chartTheme";

interface MonthPoint { month_index: number; n: number; mean_dq: number; pct_grade_a: number }
interface Props { data: MonthPoint[] }

type Metric = "mean_dq" | "pct_grade_a";
const METRIC_LABEL: Record<Metric, string> = { mean_dq: "Mean DQ score", pct_grade_a: "% Grade A" };

// Smooth line, gradient area fill, rich cross-hair tooltip that tracks touch
// as well as mouse — the "line-tooltip-touch" pattern:
// https://echarts.apache.org/examples/en/editor.html?c=line-tooltip-touch
export function DQTrendChart({ data }: Props) {
  const c = useChartColors();
  const [metric, setMetric] = useState<Metric>("mean_dq");
  const sorted = [...data].sort((a, b) => a.month_index - b.month_index);

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      triggerOn: "mousemove|click",
      axisPointer: { type: "cross", label: { backgroundColor: c["--chart-tooltip-bg"] } },
      backgroundColor: c["--chart-tooltip-bg"],
      borderColor: "transparent",
      textStyle: { color: c["--chart-tooltip-text"], fontFamily: "var(--font-geist-sans)", fontSize: 13 },
      formatter: (p: { axisValue: string; value: number }[]) =>
        `Month ${p[0].axisValue}<br/><b>${p[0].value?.toFixed(2)}</b> ${METRIC_LABEL[metric]}`,
    },
    grid: { left: 12, right: 16, top: 16, bottom: 26, containLabel: true },
    xAxis: {
      type: "category",
      boundaryGap: false,
      data: sorted.map(d => d.month_index),
      axisLabel: { color: c["--color-text-tertiary"], fontSize: 10, fontFamily: "var(--font-geist-mono)" },
      axisLine: { lineStyle: { color: c["--color-border"] } },
      axisTick: { show: false },
    },
    yAxis: {
      type: "value",
      min: metric === "pct_grade_a" ? 0 : "dataMin",
      max: 100,
      axisLabel: { color: c["--color-text-tertiary"], fontSize: 10, fontFamily: "var(--font-geist-mono)" },
      splitLine: { lineStyle: { color: c["--chart-grid"] } },
    },
    series: [
      {
        type: "line",
        name: METRIC_LABEL[metric],
        data: sorted.map(d => Math.round(d[metric] * 100) / 100),
        smooth: 0.35,
        symbol: "circle",
        symbolSize: 6,
        showSymbol: false,
        lineStyle: { width: 2.5, color: c["--color-accent"] },
        itemStyle: { color: c["--color-accent"], borderColor: c["--color-bg-surface"], borderWidth: 2 },
        areaStyle: {
          color: {
            type: "linear", x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: withAlpha(c["--color-accent"], 0.33) },
              { offset: 1, color: withAlpha(c["--color-accent"], 0.01) },
            ],
          },
        },
        emphasis: { focus: "series", itemStyle: { shadowBlur: 8, shadowColor: c["--color-accent"] } },
      },
    ],
  };

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "flex-end", gap: 6, marginBottom: 8 }}>
        {(Object.keys(METRIC_LABEL) as Metric[]).map(m => (
          <button
            key={m}
            onClick={() => setMetric(m)}
            style={{
              fontSize: 11, padding: "3px 9px", borderRadius: 5, cursor: "pointer",
              border: `1px solid ${metric === m ? "var(--color-accent)" : "var(--color-border)"}`,
              background: metric === m ? "var(--color-accent-muted)" : "transparent",
              color: metric === m ? "var(--color-accent)" : "var(--color-text-tertiary)",
              transition: "all 150ms ease",
            }}
          >{METRIC_LABEL[m]}</button>
        ))}
      </div>
      <ReactECharts option={option} style={{ height: 220 }} notMerge lazyUpdate />
    </div>
  );
}
