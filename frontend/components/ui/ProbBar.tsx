"use client";

interface Props {
  value?: number | null;
  lower?: number | null;
  upper?: number | null;
  showLabel?: boolean;
  height?: number;
}

function getRiskColor(v: number): string {
  if (v >= 0.3) return "#EF4444";
  if (v >= 0.1) return "#F59E0B";
  return "#22C55E";
}

export function ProbBar({ value, lower, upper, showLabel = true, height = 4 }: Props) {
  const safe = typeof value === "number" && isFinite(value) ? value : 0;
  const safeL = typeof lower === "number" && isFinite(lower) ? lower : undefined;
  const safeU = typeof upper === "number" && isFinite(upper) ? upper : undefined;
  const color = getRiskColor(safe);
  const pct = Math.min(safe * 100, 100);

  return (
    <div>
      <div style={{
        height,
        background: "var(--color-bg-subtle)",
        borderRadius: height / 2,
        overflow: "hidden",
        position: "relative",
      }}>
        <div style={{
          height: "100%",
          width: `${pct}%`,
          background: color,
          borderRadius: height / 2,
          transition: "width 0.6s cubic-bezier(0.16,1,0.3,1)",
        }} />
        {safeL !== undefined && safeU !== undefined && (
          <div style={{
            position: "absolute",
            top: 0,
            left: `${Math.min(safeL * 100, 100)}%`,
            width: `${Math.min((safeU - safeL) * 100, 100)}%`,
            height: "100%",
            background: `${color}40`,
          }} />
        )}
      </div>
      {showLabel && (
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: 4,
          fontSize: 11,
          fontFamily: "var(--font-geist-mono), monospace",
          color: "var(--color-text-secondary)",
          fontVariantNumeric: "tabular-nums",
        }}>
          <span style={{ color }}>{safe.toFixed(3)}</span>
          {safeL !== undefined && safeU !== undefined && (
            <span style={{ color: "var(--color-text-tertiary)" }}>
              [{safeL.toFixed(3)} – {safeU.toFixed(3)}]
            </span>
          )}
        </div>
      )}
    </div>
  );
}
