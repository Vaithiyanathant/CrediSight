"use client";
import { useTheme } from "@/lib/theme";
import { useMemo } from "react";

// Canvas (what ECharts paints into) cannot resolve CSS custom properties —
// `itemStyle: { color: "var(--x)" }` silently renders as black. Charts need
// the actual computed color string, re-read whenever the theme toggles.
const TOKENS = [
  "--color-bg-surface", "--color-border", "--color-text-primary", "--color-text-secondary",
  "--color-text-tertiary", "--color-accent", "--color-risk-high", "--color-risk-medium",
  "--color-risk-low", "--color-survival", "--chart-1", "--chart-2", "--chart-3", "--chart-4",
  "--chart-5", "--chart-6", "--chart-7", "--chart-8", "--chart-grid", "--chart-tooltip-bg",
  "--chart-tooltip-text", "--color-sheen",
] as const;

type Token = (typeof TOKENS)[number];
export type ChartColors = Record<Token, string>;

function readTokens(): ChartColors {
  if (typeof document === "undefined") {
    // SSR fallback — dark defaults, overwritten on client mount.
    return Object.fromEntries(TOKENS.map(t => [t, "#71717A"])) as ChartColors;
  }
  const style = getComputedStyle(document.documentElement);
  return Object.fromEntries(TOKENS.map(t => [t, style.getPropertyValue(t).trim() || "#71717A"])) as ChartColors;
}

/** Live, theme-aware chart palette. Re-reads computed CSS whenever `theme` flips. */
export function useChartColors(): ChartColors {
  const { theme } = useTheme();
  // eslint-disable-next-line react-hooks/exhaustive-deps -- `theme` is the re-read trigger
  return useMemo(() => readTokens(), [theme]);
}

export const STATE_CHART_KEYS = ["chart-1","chart-2","chart-3","chart-4","chart-5","chart-6","chart-7","chart-8"] as const;

/**
 * Apply alpha to a resolved token. Tokens here are `hsl(h s% l%)` strings
 * (modern space syntax) — `${color}55` (the hex-alpha-suffix trick) produces
 * an invalid color string on an hsl() value, so this uses the hsl slash-alpha
 * syntax instead: `hsl(h s% l% / a)`.
 */
export function withAlpha(color: string, alpha: number): string {
  if (color.startsWith("hsl(") && color.endsWith(")")) {
    return `${color.slice(0, -1)} / ${alpha})`;
  }
  return color; // already rgba()/hex fallback — used as-is
}
