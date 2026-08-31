"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChartBar, Warning, Flask, Brain, ShieldCheck } from "@phosphor-icons/react";

const MOBILE_NAV = [
  { href: "/portfolio",  icon: ChartBar,   label: "Portfolio" },
  { href: "/anomalies",  icon: Warning,    label: "Anomalies" },
  { href: "/scenarios",  icon: Flask,      label: "Scenarios" },
  { href: "/copilot",    icon: Brain,      label: "Copilot" },
  { href: "/dq",         icon: ShieldCheck,label: "DQ" },
];

export function MobileNav() {
  const pathname = usePathname();

  return (
    <nav style={{
      position: "fixed",
      bottom: 0,
      left: 0,
      right: 0,
      background: "var(--color-bg-surface)",
      borderTop: "1px solid var(--color-border)",
      zIndex: 40,
      paddingBottom: "env(safe-area-inset-bottom)",
    }} className="mobile-nav-bar">
      {MOBILE_NAV.map(({ href, icon: Icon, label }) => {
        const isActive = pathname === href || pathname.startsWith(href + "/");
        return (
          <Link
            key={href}
            href={href}
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 4,
              padding: "10px 4px",
              color: isActive ? "var(--color-accent)" : "var(--color-text-tertiary)",
              textDecoration: "none",
              fontSize: 10,
              fontWeight: isActive ? 500 : 400,
              transition: "color 150ms ease",
            }}
          >
            <Icon size={20} weight={isActive ? "fill" : "regular"} />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
