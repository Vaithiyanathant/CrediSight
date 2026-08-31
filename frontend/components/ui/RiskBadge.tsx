"use client";

import { WarningCircle, CheckCircle, Info, XCircle } from "@phosphor-icons/react";

type Level = "low" | "medium" | "high" | "neutral" | "ERROR" | "WARNING" | "INFO";

interface Props {
  level: Level;
  label?: string;
  size?: "sm" | "md";
}

const CONFIG: Record<Level, { bg: string; text: string; border: string; Icon: React.ElementType }> = {
  low:     { bg: "rgba(34,197,94,0.1)",   text: "#22C55E", border: "rgba(34,197,94,0.2)",   Icon: CheckCircle },
  medium:  { bg: "rgba(245,158,11,0.1)",  text: "#F59E0B", border: "rgba(245,158,11,0.2)",  Icon: WarningCircle },
  high:    { bg: "rgba(239,68,68,0.1)",   text: "#EF4444", border: "rgba(239,68,68,0.2)",   Icon: XCircle },
  neutral: { bg: "rgba(99,102,241,0.1)",  text: "#6366F1", border: "rgba(99,102,241,0.2)",  Icon: Info },
  ERROR:   { bg: "rgba(239,68,68,0.1)",   text: "#EF4444", border: "rgba(239,68,68,0.2)",   Icon: XCircle },
  WARNING: { bg: "rgba(245,158,11,0.1)",  text: "#F59E0B", border: "rgba(245,158,11,0.2)",  Icon: WarningCircle },
  INFO:    { bg: "rgba(99,102,241,0.1)",  text: "#6366F1", border: "rgba(99,102,241,0.2)",  Icon: Info },
};

export function RiskBadge({ level, label, size = "md" }: Props) {
  const cfg = CONFIG[level] ?? CONFIG.neutral;
  const Icon = cfg.Icon;
  const displayLabel = label ?? level;
  const iconSize = size === "sm" ? 12 : 14;
  const fontSize = size === "sm" ? "10px" : "11px";
  const px = size === "sm" ? "6px" : "8px";
  const py = size === "sm" ? "1px" : "2px";

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "4px",
        background: cfg.bg,
        color: cfg.text,
        border: `1px solid ${cfg.border}`,
        borderRadius: "4px",
        padding: `${py} ${px}`,
        fontFamily: "var(--font-geist-mono), monospace",
        fontSize,
        fontWeight: 500,
        letterSpacing: "0.07em",
        textTransform: "uppercase",
        fontVariantNumeric: "tabular-nums",
        whiteSpace: "nowrap",
      }}
    >
      <Icon size={iconSize} weight="fill" />
      {displayLabel}
    </span>
  );
}
