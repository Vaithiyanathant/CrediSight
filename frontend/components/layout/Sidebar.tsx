"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ChartBar,
  MagnifyingGlass,
  Warning,
  Flask,
  Brain,
  ShieldCheck,
  FileText,
  Pulse,
  ArrowsLeftRight,
} from "@phosphor-icons/react";

const NAV = [
  { href: "/portfolio",     icon: ChartBar,        label: "Portfolio Risk" },
  { href: "/loan",          icon: MagnifyingGlass,  label: "Loan 360" },
  { href: "/anomalies",     icon: Warning,          label: "Anomaly Review" },
  { href: "/scenarios",     icon: Flask,            label: "Scenario Studio" },
  { href: "/copilot",       icon: Brain,            label: "AI Copilot" },
  { href: "/explainability",icon: Pulse,            label: "Explainability" },
  { href: "/dq",            icon: ShieldCheck,      label: "Data Quality" },
  { href: "/drift",         icon: ArrowsLeftRight,  label: "Drift Monitor" },
  { href: "/submission",    icon: FileText,         label: "Submission" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="desktop-sidebar" style={{
      position: "fixed",
      top: 0,
      left: 0,
      bottom: 0,
      width: "var(--sidebar-width)",
      background: "var(--color-chrome-1)",
      borderRight: "1px solid var(--color-chrome-border)",
      display: "flex",
      flexDirection: "column",
      zIndex: 40,
      overflowY: "auto",
      overflowX: "hidden",
    }}>
      {/* Logo */}
      <div style={{
        height: "var(--header-height)",
        display: "flex",
        alignItems: "center",
        padding: "0 20px",
        borderBottom: "1px solid var(--color-chrome-border)",
        flexShrink: 0,
      }}>
        {/* logo.png is a marketing lockup (2172x724): icon + wordmark + tagline
            + a four-icon feature strip. At rail width the tagline and strip are
            illegible noise, and its "Credi" wordmark is dark navy — invisible
            against the dark chrome surface. So only the icon mark is used
            (pre-cropped to logo-mark.png, square and padded so it scales
            predictably), with the wordmark set as themed text that follows the
            active theme's contrast instead of being baked into a bitmap. */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/logo-mark.png"
            alt=""
            aria-hidden="true"
            style={{ width: 30, height: 30, objectFit: "contain", flexShrink: 0, display: "block" }}
          />
          <div style={{ minWidth: 0 }}>
            <p style={{ fontSize: 14, fontWeight: 600, color: "var(--color-chrome-title)", lineHeight: 1.2, margin: 0, whiteSpace: "nowrap" }}>
              CrediSight
            </p>
            <p style={{ fontSize: 10, color: "var(--color-chrome-text)", lineHeight: 1.3, margin: 0, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              Loan performance intelligence
            </p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ padding: "12px 10px", flex: 1 }}>
        <p style={{ fontSize: 10, fontWeight: 500, color: "var(--color-chrome-text)", letterSpacing: "0.1em", textTransform: "uppercase", padding: "4px 10px 10px" }}>
          Navigation
        </p>
        <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: 2 }}>
          {NAV.map(({ href, icon: Icon, label }) => {
            const isActive = pathname === href || pathname.startsWith(href + "/");
            return (
              <li key={href}>
                <Link
                  href={href}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "8px 10px",
                    borderRadius: 6,
                    fontSize: 14,
                    fontWeight: isActive ? 500 : 400,
                    // Active item = a solid accent-strong pill with a bit of
                    // lift, white text on it — not a translucent tint + left
                    // bar. That combination is what keeps it legible against
                    // both the light and dark chrome surface.
                    color: isActive ? "var(--color-text-inverse)" : "var(--color-chrome-text)",
                    background: isActive ? "var(--color-accent)" : "transparent",
                    boxShadow: isActive ? "var(--shadow-sm)" : "none",
                    textDecoration: "none",
                    transition: "background 150ms ease, color 150ms ease, box-shadow 150ms ease",
                  }}
                  onMouseEnter={e => {
                    if (!isActive) {
                      const el = e.currentTarget as HTMLElement;
                      el.style.color = "var(--color-chrome-title)";
                      el.style.background = "var(--color-chrome-hover)";
                    }
                  }}
                  onMouseLeave={e => {
                    if (!isActive) {
                      const el = e.currentTarget as HTMLElement;
                      el.style.color = "var(--color-chrome-text)";
                      el.style.background = "transparent";
                    }
                  }}
                >
                  <Icon size={16} weight={isActive ? "fill" : "regular"} color={isActive ? "var(--color-text-inverse)" : undefined} />
                  <span>{label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Footer */}
      <div style={{
        padding: "12px 20px",
        borderTop: "1px solid var(--color-chrome-border)",
        flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{
            width: 8,
            height: 8,
          }} className="gs-live-dot" />
          <span style={{ fontSize: 11, color: "var(--color-chrome-text)", fontFamily: "var(--font-geist-mono), monospace" }}>
            API Connected
          </span>
        </div>
      </div>
    </aside>
  );
}
