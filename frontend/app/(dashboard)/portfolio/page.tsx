"use client";
import { useEffect, useState, useCallback, useRef } from "react";
import { portfolioApi } from "@/lib/api";
import { KpiCard } from "@/components/ui/KpiCard";
import { RiskBadge } from "@/components/ui/RiskBadge";
import { ProbBar } from "@/components/ui/ProbBar";
import { SkeletonTable } from "@/components/ui/Skeleton";
import { PortfolioRiskChart } from "@/components/charts/PortfolioRiskChart";
import { StateDistributionChart } from "@/components/charts/StateDistributionChart";
import type { PortfolioSummary, WatchlistEntry } from "@/types";
import { ArrowsClockwise, FunnelSimple, ArrowRight, CaretUpDown, ChartPie, Stack, Wallet, WarningOctagon, TrendUp, ChartLineDown, ArrowsCounterClockwise, Fire, CheckCircle } from "@phosphor-icons/react";
import Link from "next/link";

function fmt(n:number,type:"pct"|"usd"|"num"="num"):string{
  if(type==="pct") return `${(n*100).toFixed(2)}%`;
  if(type==="usd"){ if(n>=1e9)return `$${(n/1e9).toFixed(2)}B`; if(n>=1e6)return `$${(n/1e6).toFixed(1)}M`; if(n>=1e3)return `$${(n/1e3).toFixed(0)}K`; return `$${n.toFixed(0)}`; }
  return n.toLocaleString();
}

function DQBadge({grade}:{grade:string}){
  const c:Record<string,{bg:string;text:string}>={A:{bg:"rgba(34,197,94,.1)",text:"#22C55E"},B:{bg:"rgba(132,204,22,.1)",text:"#84CC16"},C:{bg:"rgba(234,179,8,.1)",text:"#EAB308"},D:{bg:"rgba(249,115,22,.1)",text:"#F97316"},F:{bg:"rgba(239,68,68,.1)",text:"#EF4444"}};
  const s=c[grade]??{bg:"rgba(161,161,170,.1)",text:"#A1A1AA"};
  return <span style={{display:"inline-block",background:s.bg,color:s.text,borderRadius:4,padding:"1px 7px",fontSize:11,fontFamily:"var(--font-geist-mono),monospace",fontWeight:600}}>{grade}</span>;
}

export default function PortfolioPage(){
  const [summary,setSummary]=useState<PortfolioSummary|null>(null);
  const [watchlist,setWatchlist]=useState<WatchlistEntry[]>([]);
  const [total,setTotal]=useState(0);
  const [loading,setLoading]=useState(true);
  const [wloading,setWloading]=useState(true);
  const [refreshing,setRefreshing]=useState(false);
  const [error,setError]=useState<string|null>(null);
  const [sortKey,setSortKey]=useState<keyof WatchlistEntry>("prob_next_12m_default");
  const [sortDir,setSortDir]=useState<"asc"|"desc">("desc");
  const [minProb,setMinProb]=useState(0);
  // Gates the one-shot dashboard entrance choreography: the skeleton -> real
  // content swap (and with it every card-in/icon-pop/value-pop animation)
  // must happen exactly once, on the page's first successful load. A manual
  // Refresh click or a live-poll tick re-fetches in the background — `loading`
  // stays false the whole time so KpiCard/chart cards never unmount back to
  // their skeleton branch and never replay. An error->retry is the one case
  // that legitimately re-arms it, since the user is seeing a first load fail.
  const hasLoadedOnce=useRef(false);

  const load=useCallback(async()=>{
    const isFirstAttempt=!hasLoadedOnce.current;
    if(isFirstAttempt){ setLoading(true); setWloading(true); } else { setRefreshing(true); }
    setError(null);
    try{
      const [s,w]=await Promise.all([portfolioApi.summary(),portfolioApi.watchlist({n:100,min_default_prob:minProb})]);
      setSummary(s.data); setWatchlist(w.data.entries??[]); setTotal(w.data.n??w.data.entries?.length??0);
      hasLoadedOnce.current=true;
    }catch(e:unknown){
      setError(e instanceof Error?e.message:"Failed to load");
      // Only a failed FIRST load stays eligible to replay the cascade on
      // retry. A refresh that fails after good data is already showing keeps
      // hasLoadedOnce true — the page has data, it just couldn't update it.
      if(isFirstAttempt) hasLoadedOnce.current=false;
    }
    finally{ setLoading(false); setWloading(false); setRefreshing(false); }
  },[minProb]);

  useEffect(()=>{load();},[load]);

  const sorted=[...watchlist].sort((a,b)=>{
    const av=a[sortKey] as number|string, bv=b[sortKey] as number|string;
    const cmp=av<bv?-1:av>bv?1:0; return sortDir==="desc"?-cmp:cmp;
  });

  const handleSort=(k:keyof WatchlistEntry)=>{ if(k===sortKey)setSortDir(d=>d==="asc"?"desc":"asc"); else{setSortKey(k);setSortDir("desc");} };

  const TH=({col,label}:{col:keyof WatchlistEntry;label:string})=>(
    <th onClick={()=>handleSort(col)} style={{fontSize:11,letterSpacing:"0.06em",textTransform:"uppercase",color:sortKey===col?"var(--color-accent)":"var(--color-text-tertiary)",fontWeight:400,textAlign:"left",padding:"10px 12px",borderBottom:"1px solid var(--color-border)",cursor:"pointer",whiteSpace:"nowrap",userSelect:"none"}}>
      <span style={{display:"inline-flex",alignItems:"center",gap:4}}>{label}<CaretUpDown size={11}/></span>
    </th>
  );

  return(
    <div className="animate-fade-up" style={{maxWidth:1400,margin:"0 auto"}}>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:28}}>
        <p style={{fontSize:13,color:"var(--color-text-tertiary)",fontFamily:"var(--font-geist-mono),monospace"}}>{summary?`Month ${summary.as_of_month_index} · ${summary.total_loans.toLocaleString()} loans`:"Loading..."}</p>
        <button onClick={load} disabled={loading||refreshing} style={{display:"flex",alignItems:"center",gap:6,padding:"7px 14px",borderRadius:6,background:"transparent",border:"1px solid var(--color-border)",color:"var(--color-text-secondary)",fontSize:13,cursor:"pointer"}}>
          <ArrowsClockwise size={14} style={refreshing?{animation:"icon-spin .8s linear infinite"}:undefined}/>{refreshing?"Refreshing…":"Refresh"}
        </button>
      </div>
      {error&&<div style={{background:"rgba(239,68,68,.08)",border:"1px solid rgba(239,68,68,.2)",borderRadius:6,padding:"12px 16px",marginBottom:20,color:"#EF4444",fontSize:13}}>{error}</div>}
      <div className="gs-panel" style={{marginBottom:24}}>
        <div className="gs-panel__head">
          <p className="gs-panel__title">Portfolio overview</p>
          <span style={{display:"inline-flex",alignItems:"center",gap:6,fontSize:11,fontWeight:600,padding:"3px 10px",borderRadius:999,background:"hsl(142 65% 52% / .13)",border:"1px solid hsl(142 65% 52% / .30)",color:"var(--color-risk-low)"}}>
            <span className="gs-live-dot"/>Live
          </span>
        </div>
        <div data-kpi-row style={{gridTemplateColumns:"repeat(4,1fr)",gap:14,marginBottom:14}}>
          <KpiCard label="Total loans" icon={Stack} tone="running" value={loading?"—":fmt(summary?.total_loans??0)} chip={loading?undefined:`${fmt(summary?.active_loans??0)} active`} chipIcon={CheckCircle} sub={loading?undefined:`${fmt(summary?.terminal_loans??0)} terminal`} loading={loading}/>
          <KpiCard label="Portfolio balance" icon={Wallet} tone="running" value={loading?"—":fmt(summary?.total_balance??0,"usd")} sub={loading?undefined:"across every loan"} loading={loading}/>
          <KpiCard label="12m default rate" icon={WarningOctagon} tone={(summary?.projected_default_rate??0)>=.15?"failed":(summary?.projected_default_rate??0)>=.05?"wait":"completed"} value={loading?"—":fmt(summary?.projected_default_rate??0,"pct")} chip={loading?undefined:`${fmt(summary?.risk_distribution?.high??0)} high risk`} chipIcon={WarningOctagon} loading={loading}/>
          <KpiCard label="Delinquency rate" icon={TrendUp} tone={(summary?.delinquency_rate??0)>=.1?"failed":"wait"} value={loading?"—":fmt(summary?.delinquency_rate??0,"pct")} sub={loading?undefined:"of active loans"} loading={loading}/>
        </div>
        <div data-kpi-row style={{gridTemplateColumns:"repeat(4,1fr)",gap:14}}>
          {/* expected_loss_pct_of_balance is already a percentage (0.29 = 0.29%),
              not a 0-1 fraction — do not route it through fmt("pct")'s *100. */}
          <KpiCard label="Expected loss rate" icon={ChartLineDown} tone="wait" value={loading?"—":`${(summary?.expected_loss_pct_of_balance??0).toFixed(2)}%`} sub={loading?undefined:`${fmt(summary?.expected_loss??0,"usd")} total`} loading={loading}/>
          <KpiCard label="Prepayment 12m" icon={ArrowsCounterClockwise} tone="running" value={loading?"—":fmt(summary?.projected_prepayment_rate??0,"pct")} sub={loading?undefined:"projected"} loading={loading}/>
          <KpiCard label="High risk" icon={Fire} tone="failed" value={loading?"—":fmt(summary?.risk_distribution?.high??0)} chip={loading?undefined:`${fmt(summary?.risk_distribution?.medium??0)} medium`} chipIcon={WarningOctagon} loading={loading}/>
          <KpiCard label="Terminal" icon={CheckCircle} tone="completed" value={loading?"—":fmt(summary?.terminal_loans??0)} chip={loading?undefined:"prepaid or closed"} chipIcon={CheckCircle} loading={loading}/>
        </div>
      </div>
      <div className="gs-chart-row" style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20,marginBottom:32}}>
        <div className="gs-widget" style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:16}}><ChartPie size={16} color="var(--color-accent)"/><p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)"}}>Risk Distribution</p></div>
          {loading?<div className="skeleton skel-chart"/>:<PortfolioRiskChart data={summary?.risk_distribution??{low:0,medium:0,high:0}}/>}
        </div>
        <div className="gs-widget" style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:16}}><ChartPie size={16} color="var(--color-survival)"/><p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)"}}>Loan State Distribution</p></div>
          {loading?<div className="skeleton skel-chart"/>:<StateDistributionChart data={summary?.state_distribution??{}} onRefresh={load}/>}
        </div>
      </div>
      <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8}}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"16px 24px",borderBottom:"1px solid var(--color-border)"}}>
          <div><p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)"}}>Watchlist</p><p style={{fontSize:12,color:"var(--color-text-tertiary)",marginTop:2}}>{total} loans</p></div>
          <div style={{display:"flex",alignItems:"center",gap:8}}>
            <FunnelSimple size={14} color="var(--color-text-tertiary)"/>
            <select value={minProb} onChange={e=>setMinProb(Number(e.target.value))} style={{background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)",borderRadius:5,padding:"5px 8px",fontSize:12,color:"var(--color-text-secondary)",cursor:"pointer"}}>
              <option value={0}>All risk</option><option value={0.1}>Medium+ (10%)</option><option value={0.3}>High (30%)</option>
            </select>
          </div>
        </div>
        <div style={{overflowX:"auto"}}>
          {wloading?<div style={{padding:"20px 24px"}}><SkeletonTable rows={8} cols={7}/></div>:sorted.length===0?<div style={{padding:"40px 24px",textAlign:"center",color:"var(--color-text-tertiary)",fontSize:14}}>No loans match filter.</div>:(
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:13}}>
            <thead><tr><TH col="loan_id" label="Loan ID"/><TH col="current_status" label="Status"/><TH col="current_balance" label="Balance"/><TH col="prob_next_12m_default" label="12m Default"/><TH col="anomaly_score" label="Anomaly"/><th style={{fontSize:11,letterSpacing:"0.06em",textTransform:"uppercase",color:"var(--color-text-tertiary)",fontWeight:400,padding:"10px 12px",borderBottom:"1px solid var(--color-border)"}}>DQ</th><th style={{fontSize:11,letterSpacing:"0.06em",textTransform:"uppercase",color:"var(--color-text-tertiary)",fontWeight:400,padding:"10px 12px",borderBottom:"1px solid var(--color-border)"}}>Top Driver</th><th style={{padding:"10px 12px",borderBottom:"1px solid var(--color-border)"}}></th></tr></thead>
            <tbody>{sorted.slice(0,50).map((row,i)=>(
              <tr key={row.loan_id} style={{borderBottom:"1px solid var(--color-border)",transition:"background 100ms ease"}} onMouseEnter={e=>{(e.currentTarget as HTMLElement).style.background="var(--color-bg-elevated)";}} onMouseLeave={e=>{(e.currentTarget as HTMLElement).style.background="transparent";}}>
                <td style={{padding:"10px 12px",fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:"var(--color-text-primary)"}}>{row.loan_id}</td>
                <td style={{padding:"10px 12px"}}><RiskBadge level={row.current_status==="Default"||row.current_status==="90DPD"?"high":row.current_status==="60DPD"||row.current_status==="30DPD"?"medium":row.current_status==="Prepaid"||row.current_status==="Closed"?"neutral":"low"} label={row.current_status} size="sm"/></td>
                <td style={{padding:"10px 12px",fontFamily:"var(--font-geist-mono),monospace",textAlign:"right",color:"var(--color-text-secondary)"}}>{fmt(row.current_balance,"usd")}</td>
                <td style={{padding:"10px 12px",minWidth:120}}><ProbBar value={row.prob_next_12m_default} height={3}/></td>
                <td style={{padding:"10px 12px",fontFamily:"var(--font-geist-mono),monospace",textAlign:"right"}}><span style={{color:(row.anomaly_score??0)>=.6?"#EF4444":(row.anomaly_score??0)>=.3?"#F59E0B":"#22C55E"}}>{(row.anomaly_score??0).toFixed(3)}</span></td>
                <td style={{padding:"10px 12px"}}><DQBadge grade={row.dq_grade??""}/></td>
                <td style={{padding:"10px 12px",fontSize:12,color:"var(--color-text-tertiary)",maxWidth:140,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{row.top_drivers?.[0] ?? row.top_driver_1 ?? "—"}</td>
                <td style={{padding:"10px 12px"}}><Link href={`/loan/${row.loan_id}`} style={{display:"inline-flex",alignItems:"center",gap:4,fontSize:12,color:"var(--color-accent)",textDecoration:"none"}}>View<ArrowRight size={12}/></Link></td>
              </tr>
            ))}</tbody>
          </table>
          )}
        </div>
      </div>
    </div>
  );
}
