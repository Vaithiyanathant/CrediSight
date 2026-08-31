"use client";

import ReactECharts from "echarts-for-react";
import type { SurvivalPoint } from "@/types";

interface Props {
  curve: SurvivalPoint[];
  title?: string;
}

export function SurvivalChart({ curve, title }: Props) {
  const option = {
    backgroundColor: "transparent",
    title: title
      ? { text: title, textStyle: { color: "#A1A1AA", fontSize: 12, fontFamily: "var(--font-geist-sans)", fontWeight: 400 }, top: 0 }
      : undefined,
    tooltip: {
      trigger: "axis",
      backgroundColor: "#1A1A1F",
      borderColor: "#2E2E36",
      textStyle: { color: "#FAFAFA", fontFamily: "var(--font-geist-mono)", fontSize: 12 },
      formatter: (params: Array<{ axisValue: number; value: number; seriesName: string; color: string }>) => {
        const t = params[0]?.axisValue;
        const lines = params.map(p => `<span style="color:${p.color}">&#9632;</span> ${p.seriesName}: ${p.value?.toFixed(4)}`);
        return `t=${t}m<br/>${lines.join("<br/>")}`;
      },
    },
    legend: {
      bottom: 0,
      textStyle: { color: "#A1A1AA", fontSize: 11 },
      icon: "roundRect",
    },
    grid: { left: 12, right: 16, top: title ? 28 : 8, bottom: 36, containLabel: true },
    xAxis: {
      type: "category",
      data: curve.map(p => p.t),
      name: "Months",
      nameTextStyle: { color: "#71717A", fontSize: 10 },
      axisLabel: { color: "#71717A", fontSize: 10, fontFamily: "var(--font-geist-mono)" },
      axisLine: { lineStyle: { color: "#2E2E36" } },
      axisTick: { show: false },
    },
    yAxis: {
      type: "value",
      min: 0,
      max: 1,
      axisLabel: { color: "#71717A", fontSize: 10, fontFamily: "var(--font-geist-mono)", formatter: (v: number) => v.toFixed(2) },
      splitLine: { lineStyle: { color: "#2E2E3630" } },
    },
    series: [
      {
        name: "Survival",
        type: "line",
        data: curve.map(p => p.survival),
        smooth: true,
        lineStyle: { color: "#06B6D4", width: 2 },
        itemStyle: { color: "#06B6D4" },
        symbol: "none",
        areaStyle: { color: "rgba(6,182,212,0.06)" },
      },
      {
        name: "CIF Default",
        type: "line",
        data: curve.map(p => p.cif_default),
        smooth: true,
        lineStyle: { color: "#EF4444", width: 2, type: "dashed" },
        itemStyle: { color: "#EF4444" },
        symbol: "none",
      },
      {
        name: "CIF Prepay",
        type: "line",
        data: curve.map(p => p.cif_prepay),
        smooth: true,
        lineStyle: { color: "#6366F1", width: 2, type: "dashed" },
        itemStyle: { color: "#6366F1" },
        symbol: "none",
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: 260 }} />;
}
