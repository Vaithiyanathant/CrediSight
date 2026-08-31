"use client";

import ReactECharts from "echarts-for-react";
import { useChartColors, withAlpha } from "@/lib/chartTheme";

interface Props {
  data: { low: number; medium: number; high: number };
}

/**
 * Risk-mix half-donut (gauge style).
 *
 * Pattern: https://echarts.apache.org/examples/en/editor.html?c=pie-half-donut
 * — `startAngle: 180 / endAngle: 360` draws only the upper semicircle, and the
 * centre is pushed down so the arc sits above its own readout.
 *
 * Deliberately NOT `roseType`. A rose pie encodes value as *radius*, which
 * only reads well when categories are within roughly an order of magnitude.
 * This portfolio is ~9,921 low / 38 medium / 41 high — a 260:1 spread — so
 * under roseType the two risk bands that actually matter collapsed to a
 * sub-pixel radius and the chart rendered as a single green blob.
 *
 * `minAngle` keeps every band visible at any ratio: angle still encodes the
 * true share, and the minimum guarantees the small-but-important slices stay
 * hoverable rather than vanishing.
 */
export function PortfolioRiskChart({ data }: Props) {
  const c = useChartColors();
  const low = data.low ?? 0;
  const medium = data.medium ?? 0;
  const high = data.high ?? 0;
  const total = low + medium + high;
  const atRisk = medium + high;

  const slices = [
    { value: low, name: "Low", color: c["--color-risk-low"] },
    { value: medium, name: "Medium", color: c["--color-risk-medium"] },
    { value: high, name: "High", color: c["--color-risk-high"] },
  ];

  const pct = (v: number) => (total > 0 ? (v / total) * 100 : 0);
  // Below ~0.1% a percentage reads as a rounding artefact rather than a
  // figure, so small shares get an extra digit instead of showing "0.0%".
  const fmtPct = (v: number) => {
    const p = pct(v);
    if (p === 0) return "0%";
    return p < 0.1 ? `${p.toFixed(2)}%` : `${p.toFixed(1)}%`;
  };

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      trigger: "item",
      backgroundColor: c["--chart-tooltip-bg"],
      borderColor: "transparent",
      padding: [8, 12],
      textStyle: { color: c["--chart-tooltip-text"], fontSize: 12 },
      formatter: (p: { name: string; value: number; color: string }) =>
        `<div style="font-weight:600;margin-bottom:2px">${p.name} risk</div>` +
        `<span style="color:${p.color}">●</span> ${p.value.toLocaleString()} loans · ${fmtPct(p.value)}`,
    },
    legend: {
      bottom: 4,
      left: "center",
      icon: "roundRect",
      itemWidth: 9,
      itemHeight: 9,
      itemGap: 18,
      textStyle: { color: c["--color-text-secondary"], fontSize: 12 },
      formatter: (name: string) => {
        const s = slices.find(x => x.name === name);
        return s ? `${name}  ${s.value.toLocaleString()}` : name;
      },
    },
    series: [
      {
        type: "pie",
        radius: ["52%", "80%"],
        // Pushed down so the semicircle occupies the upper area and leaves
        // room beneath it for the readout and the legend.
        center: ["50%", "72%"],
        startAngle: 180,
        endAngle: 360,
        minAngle: 3,
        avoidLabelOverlap: true,
        itemStyle: {
          borderRadius: 4,
          borderColor: c["--color-bg-surface"],
          borderWidth: 2,
        },
        label: { show: false },
        labelLine: { show: false },
        emphasis: {
          scaleSize: 5,
          itemStyle: { shadowBlur: 16, shadowColor: withAlpha(c["--color-text-primary"], 0.25) },
        },
        animationType: "scale",
        animationEasing: "cubicOut",
        animationDuration: 700,
        animationDelay: (idx: number) => idx * 110,
        data: slices.map(s => ({ value: s.value, name: s.name, itemStyle: { color: s.color } })),
      },
    ],
    // Readout sits inside the open half of the donut. `title` (not `graphic`)
    // because a graphic text element anchors its LEFT edge at `left`, so
    // centring it would need hand-tuning per value length and a longer number
    // would overflow the arc; `title` honours textAlign about its own anchor.
    // Caption is kept short on purpose: the readout sits inside the donut's
    // inner hole, and a longer string ("N at medium or high risk") is wider
    // than the hole at this radius, so it rendered clipped behind the arc.
    title: {
      text: total.toLocaleString(),
      subtext: `${atRisk.toLocaleString()} at risk`,
      left: "50%",
      top: "58%",
      textAlign: "center",
      textStyle: {
        color: c["--color-text-primary"],
        fontSize: 24,
        fontWeight: 600,
        fontFamily: "var(--font-geist-mono), monospace",
      },
      subtextStyle: { color: c["--color-text-tertiary"], fontSize: 11, align: "center" },
      itemGap: 3,
    },
  };

  return <ReactECharts option={option} style={{ height: 250 }} notMerge lazyUpdate />;
}
