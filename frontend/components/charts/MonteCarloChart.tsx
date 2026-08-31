"use client";

import ReactECharts from "echarts-for-react";
import type { MonteCarloPath } from "@/types";

interface Props {
  paths?: MonteCarloPath[] | null;
  scenarioName?: string;
  color?: string;
}

export function MonteCarloChart({ paths, scenarioName, color = "#6366F1" }: Props) {
  const safePaths = Array.isArray(paths) && paths.length > 0 ? paths : [];

  if (safePaths.length === 0) {
    return (
      <div style={{ height: 280, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-text-tertiary)", fontSize: 13 }}>
        No path data available for this scenario.
      </div>
    );
  }

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      backgroundColor: "#1A1A1F",
      borderColor: "#2E2E36",
      textStyle: { color: "#FAFAFA", fontFamily: "var(--font-geist-mono)", fontSize: 12 },
      formatter: (params: Array<{ axisValue: number; value: number; seriesName: string }>) => {
        const t = params[0]?.axisValue;
        const lines = params.map(p => `${p.seriesName}: ${(p.value * 100).toFixed(2)}%`);
        return `Month ${t}<br/>${lines.join("<br/>")}`;
      },
    },
    legend: {
      bottom: 0,
      textStyle: { color: "#A1A1AA", fontSize: 11 },
      data: ["p5", "p50", "p95"],
    },
    grid: { left: 12, right: 12, top: 8, bottom: 36, containLabel: true },
    xAxis: {
      type: "category",
      data: safePaths.map(p => p.t),
      name: "Month",
      nameTextStyle: { color: "#71717A", fontSize: 10 },
      axisLabel: { color: "#71717A", fontSize: 10, fontFamily: "var(--font-geist-mono)" },
      axisLine: { lineStyle: { color: "#2E2E36" } },
      axisTick: { show: false },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "#71717A", fontSize: 10, fontFamily: "var(--font-geist-mono)", formatter: (v: number) => `${(v * 100).toFixed(0)}%` },
      splitLine: { lineStyle: { color: "#2E2E3630" } },
    },
    series: [
      {
        name: "p95",
        type: "line",
        data: safePaths.map(p => p.p95),
        smooth: true,
        symbol: "none",
        lineStyle: { width: 0 },
        itemStyle: { color },
        areaStyle: { color: `${color}18`, origin: "auto" },
        stack: "confidence",
      },
      {
        name: "p5",
        type: "line",
        data: safePaths.map(p => p.p5),
        smooth: true,
        symbol: "none",
        lineStyle: { width: 0 },
        itemStyle: { color },
        areaStyle: { color: "rgba(0,0,0,0)", origin: "auto" },
        stack: "confidence",
      },
      {
        name: "p50",
        type: "line",
        data: safePaths.map(p => p.p50),
        smooth: true,
        symbol: "none",
        lineStyle: { color, width: 2.5 },
        itemStyle: { color },
      },
    ],
  };

  return (
    <div>
      {scenarioName && (
        <p style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 8 }}>
          Scenario: <span style={{ color: "var(--color-text-primary)", fontWeight: 500 }}>{scenarioName}</span>
        </p>
      )}
      <ReactECharts option={option} style={{ height: 280 }} />
    </div>
  );
}
