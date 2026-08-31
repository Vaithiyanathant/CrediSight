"use client";

import { Warning, Robot } from "@phosphor-icons/react";

interface Props {
  verdict?: string;
  model?: string;
  latencyMs?: number;
  children?: React.ReactNode;
}

export function AiBanner({ verdict, model, latencyMs, children }: Props) {
  return (
    <div style={{
      background: "var(--color-ai-banner)",
      border: "1px solid #92400E",
      borderRadius: "6px",
      padding: "8px 12px",
      display: "flex",
      gap: "8px",
      alignItems: "flex-start",
    }}>
      <Warning size={14} weight="fill" style={{ color: "var(--color-ai-banner-text)", flexShrink: 0, marginTop: 1 }} />
      <div style={{ flex: 1 }}>
        <p style={{ fontSize: 11, fontWeight: 600, color: "var(--color-ai-banner-text)", letterSpacing: "0.06em", marginBottom: 2 }}>
          RECOMMENDATION, NOT DECISION
        </p>
        {children && (
          <p style={{ fontSize: 12, color: "var(--color-ai-banner-text)", lineHeight: 1.5 }}>
            {children}
          </p>
        )}
        {(model || verdict || latencyMs !== undefined) && (
          <div style={{ display: "flex", gap: 12, marginTop: 4, fontSize: 11, color: "#FCD34D", fontFamily: "var(--font-geist-mono), monospace" }}>
            {verdict && <span>Verifier: {verdict}</span>}
            {model && (
              <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
                <Robot size={11} />
                {model}
              </span>
            )}
            {latencyMs !== undefined && <span>{latencyMs.toFixed(0)}ms</span>}
          </div>
        )}
      </div>
    </div>
  );
}
