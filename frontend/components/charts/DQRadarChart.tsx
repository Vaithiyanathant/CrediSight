"use client";

import ReactECharts from "echarts-for-react";

interface Props {
  dimensions: {
    completeness: number;
    validity: number;
    consistency: number;
    timeliness: number;
    uniqueness: number;
    cross_source: number;
  };
}

export function DQRadarChart({ dimensions }: Props) {
  const indicators = [
    { name: "Completeness", max: 100 },
    { name: "Validity",     max: 100 },
    { name: "Consistency",  max: 100 },
    { name: "Timeliness",   max: 100 },
    { name: "Uniqueness",   max: 100 },
    { name: "Cross-Source", max: 100 },
  ];

  const values = [
    (dimensions.completeness ?? 0) * 100,
    (dimensions.validity ?? 0) * 100,
    (dimensions.consistency ?? 0) * 100,
    (dimensions.timeliness ?? 0) * 100,
    (dimensions.uniqueness ?? 0) * 100,
    (dimensions.cross_source ?? 0) * 100,
  ];

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      backgroundColor: "#1A1A1F",
      borderColor: "#2E2E36",
      textStyle: { color: "#FAFAFA", fontFamily: "var(--font-geist-mono)", fontSize: 12 },
    },
    radar: {
      indicator: indicators,
      shape: "polygon",
      splitNumber: 4,
      center: ["50%", "54%"],
      radius: "70%",
      axisName: { color: "#A1A1AA", fontSize: 11, fontFamily: "var(--font-geist-sans)" },
      splitLine: { lineStyle: { color: "#2E2E36" } },
      splitArea: { areaStyle: { color: ["#1A1A1F30", "#1A1A1F20", "#1A1A1F10", "#1A1A1F08"] } },
      axisLine: { lineStyle: { color: "#2E2E36" } },
    },
    series: [
      {
        type: "radar",
        data: [
          {
            value: values,
            name: "DQ Score",
            itemStyle: { color: "#6366F1" },
            lineStyle: { color: "#6366F1", width: 2 },
            areaStyle: { color: "rgba(99,102,241,0.12)" },
          },
        ],
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: 260 }} />;
}
