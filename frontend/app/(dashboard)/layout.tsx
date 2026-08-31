import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { MobileNav } from "@/components/layout/MobileNav";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", minHeight: "100dvh", background: "var(--color-bg-base)" }}>
      <Sidebar />
      <div style={{
        marginLeft: "var(--sidebar-width)",
        flex: 1,
        display: "flex",
        flexDirection: "column",
        minWidth: 0,
      }}>
        <Header />
        <main className="dashboard-main" style={{
          marginTop: "var(--header-height)",
          padding: "32px 32px",
          flex: 1,
          minWidth: 0,
        }}>
          {children}
        </main>
      </div>
      <MobileNav />
    </div>
  );
}
