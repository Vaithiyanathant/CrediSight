"use client";

interface Props {
  width?: string | number;
  height?: string | number;
  className?: string;
  style?: React.CSSProperties;
}

export function Skeleton({ width, height = 16, className = "", style }: Props) {
  return (
    <div
      className={`skeleton ${className}`}
      style={{ width, height, ...style }}
    />
  );
}

export function SkeletonCard({ rows = 3 }: { rows?: number }) {
  return (
    <div style={{
      background: "var(--color-bg-surface)",
      border: "1px solid var(--color-border)",
      borderRadius: "8px",
      padding: "20px 24px",
    }}>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="skeleton" style={{ height: 14, width: i === 0 ? "50%" : i === rows - 1 ? "30%" : "80%", marginBottom: i < rows - 1 ? 10 : 0 }} />
      ))}
    </div>
  );
}

export function SkeletonTable({ rows = 5, cols = 5 }: { rows?: number; cols?: number }) {
  return (
    <div style={{ overflow: "hidden" }}>
      <div style={{ display: "flex", gap: 12, paddingBottom: 12, borderBottom: "1px solid var(--color-border)", marginBottom: 8 }}>
        {Array.from({ length: cols }).map((_, i) => (
          <div key={i} className="skeleton" style={{ height: 10, flex: 1 }} />
        ))}
      </div>
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} style={{ display: "flex", gap: 12, paddingBlock: 10, borderBottom: "1px solid var(--color-border)" }}>
          {Array.from({ length: cols }).map((_, c) => (
            <div key={c} className="skeleton" style={{ height: 13, flex: 1, opacity: 1 - r * 0.08 }} />
          ))}
        </div>
      ))}
    </div>
  );
}
