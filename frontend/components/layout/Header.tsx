"use client";

import { usePathname } from "next/navigation";
import { Bell, MagnifyingGlass, Sun, Moon } from "@phosphor-icons/react";
import { useTheme } from "@/lib/theme";

const PAGE_LABELS: Record<string, string> = {
  "/portfolio":     "Portfolio Risk",
  "/loan":          "Loan 360",
  "/anomalies":     "Anomaly Review",
  "/scenarios":     "Scenario Studio",
  "/copilot":       "AI Copilot",
  "/explainability":"Explainability",
  "/dq":            "Data Quality",
  "/drift":         "Drift Monitor",
  "/submission":    "Submission",
};

export function Header() {
  const pathname = usePathname();
  const base = "/" + (pathname.split("/")[1] || "");
  const title = PAGE_LABELS[base] ?? "Dashboard";
  const { theme, toggleTheme } = useTheme();

  const btnStyle: React.CSSProperties = {
    width: 32,
    height: 32,
    borderRadius: 6,
    background: "transparent",
    border: "1px solid var(--color-border)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    color: "var(--color-text-secondary)",
    transition: "all 150ms ease",
  };

  return (
    <header style={{
      position: "fixed",
      top: 0,
      left: "var(--sidebar-width)",
      right: 0,
      height: "var(--header-height)",
      // Was a hardcoded rgba(9,9,11,...) — the pre-port dark literal that never
      // read the token system, so the header stayed black-ish regardless of
      // which theme/palette was active. Blends the theme's own chrome surface
      // instead, so light/dark and any future palette swap both apply here.
      background: "rgba(var(--color-chrome-1-rgb), 0.85)",
      backdropFilter: "blur(12px)",
      WebkitBackdropFilter: "blur(12px)",
      borderBottom: "1px solid var(--color-chrome-border)",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      paddingInline: 24,
      zIndex: 30,
      transition: "background 200ms ease, border-color 200ms ease",
    }}>
      <h1 style={{ fontSize: 16, fontWeight: 600, color: "var(--color-text-primary)" }}>
        {title}
      </h1>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>

        {/* Search */}
        <button aria-label="Search" style={btnStyle}
          onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = "var(--color-bg-elevated)"; }}
          onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}>
          <MagnifyingGlass size={15} />
        </button>

        {/* Theme Toggle */}
        <button
          aria-label="Toggle theme"
          onClick={toggleTheme}
          title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
          style={btnStyle}
          onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = "var(--color-bg-elevated)"; }}
          onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
        >
          {theme === "dark"
            ? <Sun size={15} weight="fill" color="#F59E0B" />
            : <Moon size={15} weight="fill" color="#6366F1" />
          }
        </button>

        {/* Notifications */}
        <button aria-label="Notifications" style={{ ...btnStyle, position: "relative" }}
          onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = "var(--color-bg-elevated)"; }}
          onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}>
          <Bell size={15} />
          <span style={{
            position: "absolute", top: 5, right: 5,
            width: 6, height: 6,
            background: "#EF4444",
            borderRadius: "50%",
            border: "1px solid var(--color-bg-base)",
          }} />
        </button>

        {/* User Avatar */}
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          paddingLeft: 8,
          borderLeft: "1px solid var(--color-border)",
        }}>
          <div style={{
            width: 28, height: 28, borderRadius: "50%",
            background: "var(--color-accent)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 12, fontWeight: 600, color: "#fff",
          }}>
            RV
          </div>
          <div>
            <p style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-primary)", lineHeight: 1.2 }}>Reviewer</p>
            <p style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>Risk Analyst</p>
          </div>
        </div>
      </div>
    </header>
  );
}

