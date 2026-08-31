"use client";

import ReactECharts from "echarts-for-react";
import type { DriverBlock } from "@/types";

interface Props {
  drivers: DriverBlock[];
  baseValue?: number;
  prediction?: number;
}

export function ShapWaterfallChart({ drivers, baseValue = 0, prediction }: Props) {
  const top = drivers.slice(0, 10);

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      backgroundColor: "#1A1A1F",
      borderColor: "#2E2E36",
      textStyle: { color: "#FAFAFA", fontFamily: "var(--font-geist-mono)", fontSize: 12 },
      formatter: (params: Array<{ name: string; value: number }>) => {
        const p = params[0];
        return `<b>${p.name}</b><br/>SHAP: ${p.value > 0 ? "+" : ""}${Number(p.value).toFixed(4)}`;
      },
    },
    grid: { left: 12, right: 60, top: 8, bottom: 8, containLabel: true },
    xAxis: {
      type: "value",
      axisLabel: { color: "#71717A", fontSize: 10, fontFamily: "var(--font-geist-mono)", formatter: (v: number) => v > 0 ? `+${v.toFixed(3)}` : v.toFixed(3) },
      splitLine: { lineStyle: { color: "#2E2E3630" } },
    },
    yAxis: {
      type: "category",
      data: top.map(d => d.feature).reverse(),
      axisLabel: { color: "#A1A1AA", fontSize: 11, fontFamily: "var(--font-geist-sans)", width: 140, overflow: "truncate" },
      axisLine: { lineStyle: { color: "#2E2E36" } },
      axisTick: { show: false },
    },
    series: [
      {
        type: "bar",
        barMaxWidth: 28,
        borderRadius: 3,
        data: top.map(d => ({
          value: d.shap_value,
          itemStyle: { color: d.direction === "positive" ? "#EF4444" : "#22C55E" },
        })).reverse(),
        label: {
          show: true,
          position: "right",
          color: "#A1A1AA",
          fontFamily: "var(--font-geist-mono)",
          fontSize: 10,
          formatter: (p: { value: number }) => `${Number(p.value) > 0 ? "+" : ""}${Number(p.value).toFixed(3)}`,
        },
      },
    ],
  };

  return (
    <div>
      {prediction !== undefined && (
        <div style={{ display: "flex", gap: 20, marginBottom: 12 }}>
          {baseValue !== undefined && (
            <div>
              <p style={{ fontSize: 11, color: "var(--color-text-tertiary)", letterSpacing: "0.06em", textTransform: "uppercase" }}>Base</p>
              <p style={{ fontFamily: "var(--font-geist-mono)", fontSize: 16, fontWeight: 600, color: "var(--color-text-primary)" }}>{baseValue.toFixed(3)}</p>
            </div>
          )}
          <div>
            <p style={{ fontSize: 11, color: "var(--color-text-tertiary)", letterSpacing: "0.06em", textTransform: "uppercase" }}>Prediction</p>
            <p style={{ fontFamily: "var(--font-geist-mono)", fontSize: 16, fontWeight: 600, color: prediction >= 0.3 ? "#EF4444" : prediction >= 0.1 ? "#F59E0B" : "#22C55E" }}>{prediction.toFixed(3)}</p>
          </div>
        </div>
      )}
      <ReactECharts option={option} style={{ height: Math.max(180, top.length * 32) }} />
    </div>
  );
}
