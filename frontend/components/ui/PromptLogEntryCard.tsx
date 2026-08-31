"use client";

import { useState } from "react";
import { CaretRight, CheckCircle, WarningCircle, XCircle } from "@phosphor-icons/react";
import type { PromptLogEntry } from "@/types";

/**
 * One row of the prompt audit trail, expandable into the full exchange.
 *
 * Collapsed it shows the verdict, task and latency. Expanded it shows what was
 * actually sent (system + user prompt, retrieved context) and what came back —
 * both the model's raw first attempt and the answer the client was finally
 * served. Those two differ whenever the verifier rejected the first draft, and
 * showing them side by side is the thing that makes a rejection auditable
 * rather than just a red badge.
 */

const VERDICT = {
  PASS:        { fg: "var(--color-risk-low)",    Icon: CheckCircle },
  REGENERATED: { fg: "var(--color-risk-medium)", Icon: WarningCircle },
  FALLBACK:    { fg: "var(--color-risk-high)",   Icon: XCircle },
  FAIL:        { fg: "var(--color-risk-high)",   Icon: XCircle },
} as const;

function Block({ label, body, mono = true, tone }: { label: string; body: string; mono?: boolean; tone?: string }) {
  return (
    <div style={{ marginTop: 10 }}>
      <p style={{ fontSize: 10, letterSpacing: "0.07em", textTransform: "uppercase", color: "var(--color-text-tertiary)", marginBottom: 4 }}>
        {label}
      </p>
      <pre style={{
        margin: 0, padding: "9px 11px", borderRadius: 6,
        background: "var(--color-bg-sunken)", border: "1px solid var(--color-border)",
        color: tone ?? "var(--color-text-secondary)",
        fontSize: 11.5, lineHeight: 1.55, whiteSpace: "pre-wrap", wordBreak: "break-word",
        fontFamily: mono ? "var(--font-geist-mono), monospace" : "inherit",
        maxHeight: 260, overflowY: "auto",
      }}>{body}</pre>
    </div>
  );
}

export function PromptLogEntryCard({ entry }: { entry: PromptLogEntry }) {
  const [open, setOpen] = useState(false);
  const verdictKey = (entry.verdict ?? entry.verifier_verdict ?? "") as keyof typeof VERDICT;
  const v = VERDICT[verdictKey] ?? { fg: "var(--color-text-tertiary)", Icon: WarningCircle };
  const { Icon } = v;

  // The served answer differs from the first attempt only when the verifier
  // intervened — worth calling out explicitly rather than making the reader
  // diff two long strings by eye.
  const wasRewritten = !!entry.raw_output && !!entry.final_output && entry.raw_output !== entry.final_output;
  const hasContent = !!(entry.user_prompt || entry.final_output || entry.raw_output);

  return (
    <div style={{ borderBottom: "1px solid var(--color-border)" }}>
      <button
        onClick={() => setOpen(o => !o)}
        disabled={!hasContent}
        aria-expanded={open}
        style={{
          width: "100%", textAlign: "left", background: "transparent", border: "none",
          padding: "10px 16px", cursor: hasContent ? "pointer" : "default",
          display: "flex", alignItems: "flex-start", gap: 8,
        }}
      >
        {hasContent && (
          <CaretRight
            size={11} color="var(--color-text-tertiary)"
            style={{ marginTop: 3, flexShrink: 0, transform: open ? "rotate(90deg)" : "none", transition: "transform 150ms ease" }}
          />
        )}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
            <Icon size={12} color={v.fg} />
            <span style={{ fontSize: 11, color: v.fg, fontWeight: 600 }}>
              {entry.verdict ?? entry.verifier_verdict ?? "—"}
            </span>
            <span style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginLeft: "auto", fontFamily: "var(--font-geist-mono), monospace" }}>
              {(entry.latency_ms ?? 0).toFixed(0)}ms
            </span>
          </div>
          <p style={{ fontSize: 12, color: "var(--color-text-secondary)", margin: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {entry.question ?? entry.task ?? "—"}
          </p>
          <p style={{ fontSize: 10, color: "var(--color-text-tertiary)", margin: "2px 0 0", fontFamily: "var(--font-geist-mono), monospace" }}>
            {entry.model}
            {entry.input_tokens != null && <> · {entry.input_tokens}→{entry.output_tokens ?? 0} tok</>}
          </p>
        </div>
      </button>

      {open && hasContent && (
        <div style={{ padding: "0 16px 14px", animation: "fade-in .2s ease both" }}>
          {wasRewritten && (
            <p style={{ fontSize: 11, color: "var(--color-risk-medium)", background: "hsl(45 90% 57% / .10)", border: "1px solid hsl(45 90% 57% / .28)", borderRadius: 6, padding: "6px 9px", margin: "4px 0 0" }}>
              The verifier rejected the first draft — the served answer below differs from the model&apos;s raw output.
            </p>
          )}

          {entry.system_prompt && <Block label="System prompt" body={entry.system_prompt} />}
          {entry.user_prompt && <Block label={`User prompt (${entry.user_prompt.length} chars)`} body={entry.user_prompt} />}

          {entry.retrieved_context && entry.retrieved_context.length > 0 && (
            <Block
              label={`Retrieved context (${entry.retrieved_context.length} chunks)`}
              body={entry.retrieved_context
                .map((c, i) => `[${i + 1}] ${c.citation ?? "?"}${c.score != null ? `  score=${c.score}` : ""}\n${(c.text ?? "").trim()}`)
                .join("\n\n")}
            />
          )}

          {entry.raw_output && (
            <Block
              label={wasRewritten ? "Model raw output (rejected)" : "Model output"}
              body={entry.raw_output}
              tone={wasRewritten ? "var(--color-text-tertiary)" : undefined}
            />
          )}

          {wasRewritten && entry.final_output && (
            <Block label="Served answer (after verifier)" body={entry.final_output} tone="var(--color-text-primary)" />
          )}

          {entry.verifier_failures && entry.verifier_failures.length > 0 && (
            <Block
              label={`Verifier failures (${entry.verifier_failures.length})`}
              body={entry.verifier_failures.map(f => `• ${f.kind ?? "?"}: ${f.detail ?? ""}${f.evidence ? `\n  evidence: ${f.evidence}` : ""}`).join("\n")}
              tone="var(--color-risk-high)"
            />
          )}
        </div>
      )}
    </div>
  );
}
