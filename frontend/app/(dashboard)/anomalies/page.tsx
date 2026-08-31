"use client";
import { useEffect, useState, useCallback } from "react";
import { anomalyApi, reviewerApi } from "@/lib/api";
import { RiskBadge } from "@/components/ui/RiskBadge";
import { ProbBar } from "@/components/ui/ProbBar";
import { AiBanner } from "@/components/ui/AiBanner";
import { SkeletonTable, SkeletonCard } from "@/components/ui/Skeleton";
import type { AnomalyEntry } from "@/types";
import { Warning, MagnifyingGlass, ArrowRight, CheckCircle, XCircle, ArrowUpRight } from "@phosphor-icons/react";
import Link from "next/link";

export default function AnomaliesPage(){
  const [entries,setEntries]=useState<AnomalyEntry[]>([]);
  const [total,setTotal]=useState(0);
  const [loading,setLoading]=useState(true);
  const [selected,setSelected]=useState<AnomalyEntry|null>(null);
  const [decision,setDecision]=useState<"Confirm"|"Reject"|"Escalate"|null>(null);
  const [submitting,setSubmitting]=useState(false);
  const [submitted,setSubmitted]=useState(false);
  const [minScore,setMinScore]=useState(0);
  const [search,setSearch]=useState("");

  const load=useCallback(async()=>{
    setLoading(true);
    try{
      const r=await anomalyApi.list({limit:100,min_score:minScore});
      setEntries(r.data.entries??[]); setTotal(r.data.total_matching??r.data.entries?.length??0);
    }catch{ setEntries([]); }
    finally{ setLoading(false); }
  },[minScore]);

  useEffect(()=>{load();},[load]);

  const filtered=entries.filter(e=>!search||e.loan_id.toLowerCase().includes(search.toLowerCase()));

  const submitDecision=async()=>{
    if(!decision||!selected) return;
    setSubmitting(true);
    try{
      // Backend requires month_index >= 1; every AnomalyEntry already carries
      // its real month_index — sending a hardcoded 0 here always 422'd.
      await reviewerApi.decision({loan_id:selected.loan_id,month_index:selected.month_index,human_decision:decision,model_recommendation:selected.reviewer_action,anomaly_score:selected.anomaly_score,exception_type:selected.exception_type});
      setSubmitted(true);
    }catch(e:unknown){ console.error("reviewer decision failed", e); } finally{setSubmitting(false);}
  };

  const selectEntry=(e:AnomalyEntry)=>{ setSelected(e); setDecision(null); setSubmitted(false); };

  return(
    <div className="animate-fade-up" style={{maxWidth:1400,margin:"0 auto",display:"grid",gridTemplateColumns:"1fr 380px",gap:20,alignItems:"start"}}>
      {/* List */}
      <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8}}>
        <div style={{padding:"16px 20px",borderBottom:"1px solid var(--color-border)",display:"flex",gap:12,alignItems:"center"}}>
          <Warning size={16} color="#EF4444"/>
          <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",flex:1}}>Anomaly Queue</p>
          <span style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:"var(--color-text-tertiary)"}}>{total} total</span>
          <div style={{display:"flex",alignItems:"center",gap:6,background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)",borderRadius:6,padding:"5px 10px"}}>
            <MagnifyingGlass size={13} color="var(--color-text-tertiary)"/>
            <input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search loan ID" style={{background:"transparent",border:"none",outline:"none",fontSize:12,color:"var(--color-text-primary)",width:120}}/>
          </div>
          <select value={minScore} onChange={e=>setMinScore(Number(e.target.value))} style={{background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)",borderRadius:5,padding:"5px 8px",fontSize:12,color:"var(--color-text-secondary)"}}>
            <option value={0}>All scores</option><option value={0.3}>0.3+</option><option value={0.6}>0.6+</option>
          </select>
        </div>
        <div style={{overflowX:"auto"}}>
          {loading?<div style={{padding:"20px"}}><SkeletonTable rows={8} cols={5}/></div>:filtered.length===0?<div style={{padding:"40px",textAlign:"center",color:"var(--color-text-tertiary)",fontSize:14}}>No anomalies found.</div>:(
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:13}}>
              <thead><tr>
                {["Loan ID","Score","Tier","Status","Balance","Violations",""].map(h=>(
                  <th key={h} style={{fontSize:11,letterSpacing:"0.06em",textTransform:"uppercase",color:"var(--color-text-tertiary)",fontWeight:400,textAlign:"left",padding:"10px 12px",borderBottom:"1px solid var(--color-border)",whiteSpace:"nowrap"}}>{h}</th>
                ))}
              </tr></thead>
              <tbody>{filtered.map(e=>(
                <tr key={e.loan_id} onClick={()=>selectEntry(e)} style={{borderBottom:"1px solid var(--color-border)",cursor:"pointer",background:selected?.loan_id===e.loan_id?"var(--color-bg-elevated)":"transparent",transition:"background 100ms"}} onMouseEnter={ev=>{if(selected?.loan_id!==e.loan_id)(ev.currentTarget as HTMLElement).style.background="rgba(26,26,31,.5)";}} onMouseLeave={ev=>{if(selected?.loan_id!==e.loan_id)(ev.currentTarget as HTMLElement).style.background="transparent";}}>
                  <td style={{padding:"10px 12px",fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:"var(--color-text-primary)"}}>{e.loan_id}</td>
                  <td style={{padding:"10px 12px"}}><span style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:13,fontWeight:600,color:e.anomaly_score>=.6?"#EF4444":e.anomaly_score>=.3?"#F59E0B":"#22C55E"}}>{e.anomaly_score.toFixed(3)}</span></td>
                  <td style={{padding:"10px 12px"}}><RiskBadge level={e.anomaly_tier as "low"|"medium"|"high"} label={e.anomaly_tier} size="sm"/></td>
                  <td style={{padding:"10px 12px"}}><RiskBadge level={e.current_status==="Default"||e.current_status==="90DPD"?"high":e.current_status==="60DPD"||e.current_status==="30DPD"?"medium":"low"} label={e.current_status} size="sm"/></td>
                  <td style={{padding:"10px 12px",fontFamily:"var(--font-geist-mono),monospace",textAlign:"right",fontSize:12,color:"var(--color-text-secondary)"}}>${(e.current_balance/1000).toFixed(0)}K</td>
                  <td style={{padding:"10px 12px"}}><span style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:"var(--color-text-tertiary)"}}>{(e.rules_fired??e.rule_violations)?.length??0} rules</span></td>
                  <td style={{padding:"10px 12px"}}><ArrowRight size={13} color="var(--color-text-tertiary)"/></td>
                </tr>
              ))}</tbody>
            </table>
          )}
        </div>
      </div>

      {/* Detail panel */}
      <div style={{display:"flex",flexDirection:"column",gap:16,position:"sticky",top:"calc(var(--header-height) + 32px)"}}>
        {!selected?(
          <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"32px 24px",textAlign:"center"}}>
            <Warning size={28} color="var(--color-text-tertiary)" style={{margin:"0 auto 12px"}}/>
            <p style={{fontSize:14,color:"var(--color-text-tertiary)"}}>Select a loan to review</p>
          </div>
        ):(
          <>
            <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 20px"}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:14}}>
                <div>
                  <p style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:14,fontWeight:600,color:"var(--color-text-primary)"}}>{selected.loan_id}</p>
                  <p style={{fontSize:12,color:"var(--color-text-tertiary)",marginTop:2}}>Anomaly Score: <span style={{color:selected.anomaly_score>=.6?"#EF4444":"#F59E0B",fontFamily:"var(--font-geist-mono),monospace"}}>{selected.anomaly_score.toFixed(3)}</span></p>
                </div>
                <Link href={`/loan/${selected.loan_id}`} style={{display:"inline-flex",alignItems:"center",gap:4,fontSize:12,color:"var(--color-accent)",textDecoration:"none"}}>
                  Full View<ArrowUpRight size={12}/>
                </Link>
              </div>
              {((selected.rules_fired??selected.rule_violations)?.length??0)>0&&(
                <div style={{marginBottom:14}}>
                  <p style={{fontSize:11,color:"var(--color-text-tertiary)",letterSpacing:"0.06em",textTransform:"uppercase",marginBottom:8}}>Rule Violations</p>
                  <div style={{display:"flex",flexDirection:"column",gap:4}}>
                    {(selected.rules_fired??selected.rule_violations??[]).slice(0,5).map(v=>(
                      <div key={v} style={{display:"flex",alignItems:"center",gap:6,fontSize:12,color:"var(--color-text-secondary)"}}>
                        <XCircle size={12} color="#EF4444"/>{v}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <div style={{marginBottom:14}}>
                <p style={{fontSize:11,color:"var(--color-text-tertiary)",letterSpacing:"0.06em",textTransform:"uppercase",marginBottom:8}}>Rule Severity{selected.exception_required?" · Exception Required":""}</p>
                <ProbBar value={selected.exception_prob??selected.rule_severity??0} height={4}/>
              </div>
              <AiBanner>Model recommendation based on ensemble anomaly detection. Human decision required before any action is taken.</AiBanner>
            </div>

            <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px"}}>
              <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",marginBottom:12}}>Decision</p>
              {submitted?(
                <div style={{background:"rgba(34,197,94,.08)",border:"1px solid rgba(34,197,94,.2)",borderRadius:6,padding:"10px 12px",display:"flex",alignItems:"center",gap:8}}>
                  <CheckCircle size={16} color="#22C55E"/>
                  <span style={{fontSize:13,color:"#22C55E"}}>Decision recorded: {decision}</span>
                </div>
              ):(
                <>
                  <div style={{display:"flex",gap:8,marginBottom:10}}>
                    {(["Confirm","Reject","Escalate"] as const).map(d=>(
                      <button key={d} onClick={()=>setDecision(d)} style={{flex:1,padding:"8px 0",borderRadius:6,fontSize:12,fontWeight:500,cursor:"pointer",border:`1px solid ${decision===d?d==="Confirm"?"#22C55E":d==="Reject"?"#EF4444":"#F59E0B":"var(--color-border)"}`,background:decision===d?d==="Confirm"?"rgba(34,197,94,.1)":d==="Reject"?"rgba(239,68,68,.1)":"rgba(245,158,11,.1)":"transparent",color:decision===d?d==="Confirm"?"#22C55E":d==="Reject"?"#EF4444":"#F59E0B":"var(--color-text-secondary)",transition:"all 150ms ease"}}>{d}</button>
                    ))}
                  </div>
                  <button onClick={submitDecision} disabled={!decision||submitting} style={{width:"100%",padding:"9px 0",borderRadius:6,fontSize:13,fontWeight:500,cursor:decision?"pointer":"not-allowed",border:"none",background:decision?"var(--color-accent)":"var(--color-bg-subtle)",color:decision?"#fff":"var(--color-text-tertiary)",transition:"all 150ms ease"}}>{submitting?"Submitting...":"Record Decision"}</button>
                </>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
