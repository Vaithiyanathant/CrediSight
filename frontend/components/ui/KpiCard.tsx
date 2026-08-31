"use client";

import type { Icon as PhosphorIcon } from "@phosphor-icons/react";

export type KpiTone = "pending" | "wait" | "running" | "completed" | "failed" | "neutral";

interface Props {
  label: string;
  value: string;
  /** Muted caption sitting to the right of the chip, e.g. "10 paused". */
  sub?: string;
  accent?: string;
  loading?: boolean;
  onClick?: () => void;
  /** Bare tone-coloured glyph, drawn top-right. Not a filled badge. */
  icon?: PhosphorIcon;
  /** Drives the icon colour and the chip's tinted soft-pair. */
  tone?: KpiTone;
  /** Tinted status pill, e.g. "5 active" / "6 failed". */
  chip?: string;
  /** Small glyph inside the chip, matching the reference's status pills. */
  chipIcon?: PhosphorIcon;
  spin?: boolean;
}

// Soft-pair tints from the theme kit: ~8-12% fill, ~28% border, solid text.
const TONE: Record<KpiTone, { fg: string; bg: string; bd: string }> = {
  pending:   { fg: "var(--color-text-tertiary)", bg: "hsl(215 20% 68% / .10)", bd: "hsl(215 20% 68% / .26)" },
  wait:      { fg: "var(--color-risk-medium)",   bg: "hsl(45 90% 57% / .13)",  bd: "hsl(45 90% 57% / .30)" },
  running:   { fg: "var(--color-accent)",        bg: "hsl(200 85% 52% / .13)", bd: "hsl(200 85% 52% / .30)" },
  completed: { fg: "var(--color-risk-low)",      bg: "hsl(142 65% 52% / .13)", bd: "hsl(142 65% 52% / .30)" },
  failed:    { fg: "var(--color-risk-high)",     bg: "hsl(354 80% 65% / .13)", bd: "hsl(354 80% 65% / .30)" },
  neutral:   { fg: "var(--color-text-tertiary)", bg: "var(--color-bg-subtle)", bd: "transparent" },
};

/**
 * KPI tile — anatomy matched to the reference dashboard's tiles:
 *
 *   ┌────────────────────────────────┐
 *   │ Sources                     🌐 │  label left · tone glyph right
 *   │ 15                             │  large tabular value
 *   │ [✓ 5 active]  10 paused        │  tinted status chip · muted caption
 *   └────────────────────────────────┘
 *
 * Sentence-case label (not uppercase+tracked): uppercasing runs ~20% wider and
 * made longer labels wrap on some tiles and not others, which pushed their
 * values onto different baselines and broke the row's shared baseline.
 *
 * ONE persistent root for both loading and loaded states — only the children
 * swap. Returning two different root elements would make React unmount and
 * remount the tile on every `loading` flip, replaying the entrance
 * choreography on every refresh instead of once on first load.
 */
export function KpiCard({ label, value, sub, accent, loading, onClick, icon: Icon, tone = "neutral", chip, chipIcon: ChipIcon, spin }: Props) {
  const t = TONE[tone];
  const iconColor = accent ?? t.fg;

  return (
    <div
      className="gs-kpi"
      onClick={loading ? undefined : onClick}
      style={{
        position: "relative",
        background: "var(--color-bg-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: 8,
        padding: "16px 18px",
        overflow: "hidden",
        cursor: onClick && !loading ? "pointer" : "default",
      }}
    >
      {loading ? (
        <>
          <div className="skeleton" style={{ height: 12, width: "48%", marginBottom: 12 }} />
          <div className="skeleton" style={{ height: 30, width: "34%", marginBottom: 12 }} />
          <div className="skeleton" style={{ height: 20, width: "60%", borderRadius: 99 }} />
        </>
      ) : (
        <>
          {/* Label + glyph share the top line; the glyph is absolutely placed
              so a long label never shortens the value's own width. */}
          <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
            <p style={{
              flex: 1, minWidth: 0,
              fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)",
              lineHeight: 1.3, margin: 0,
            }}>{label}</p>
            {Icon && (
              <span className="gs-kpi__icon" style={{ flex: "none", color: iconColor, display: "flex", marginTop: -2 }}>
                <Icon size={22} style={spin ? { animation: "icon-spin 1.6s linear infinite" } : undefined} />
              </span>
            )}
          </div>

          <p className="gs-kpi__value" style={{
            fontFamily: "var(--font-geist-mono), monospace",
            fontSize: 34, fontWeight: 600, color: "var(--color-text-primary)",
            lineHeight: 1.15, margin: "2px 0 0", fontVariantNumeric: "tabular-nums",
            letterSpacing: "-0.01em",
          }}>{value}</p>

          {(chip || sub) && (
            <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 9, marginTop: 9 }}>
              {chip && (
                <span style={{
                  display: "inline-flex", alignItems: "center", gap: 5,
                  fontSize: 12, fontWeight: 600, lineHeight: 1.3,
                  padding: "3px 9px", borderRadius: 999,
                  background: t.bg, border: `1px solid ${t.bd}`, color: t.fg,
                }}>
                  {ChipIcon && <ChipIcon size={12} weight="bold" />}
                  {chip}
                </span>
              )}
              {sub && <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>{sub}</span>}
            </div>
          )}
        </>
      )}
    </div>
  );
}
