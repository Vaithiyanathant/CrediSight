"use client";
import { useEffect, useState } from "react";
import { dqApi, driftApi } from "@/lib/api";
import { DQRadarChart } from "@/components/charts/DQRadarChart";
import { DQTrendChart } from "@/components/charts/DQTrendChart";
import { SkeletonCard } from "@/components/ui/Skeleton";
import type { DQSummaryResponse, DriftResponse } from "@/types";
import { ShieldCheck, ArrowsLeftRight, WarningCircle, CheckCircle, XCircle } from "@phosphor-icons/react";

function GradeBar({grade,count,total}:{grade:string;count:number;total:number}){
  const colors:Record<string,string>={A:"#22C55E",B:"#84CC16",C:"#EAB308",D:"#F97316",F:"#EF4444"};
  const c=colors[grade]??"#6366F1";
  const pct=total>0?(count/total*100):0;
  return(
    <div style={{display:"flex",alignItems:"center",gap:10}}>
      <span style={{width:16,fontFamily:"var(--font-geist-mono),monospace",fontSize:13,fontWeight:700,color:c}}>{grade}</span>
      <div style={{flex:1,height:8,background:"var(--color-bg-subtle)",borderRadius:4,overflow:"hidden"}}>
        <div style={{height:"100%",width:`${pct}%`,background:c,borderRadius:4,transition:"width .6s cubic-bezier(0.16,1,0.3,1)"}}/>
      </div>
      <span style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:"var(--color-text-secondary)",width:40,textAlign:"right"}}>{count}</span>
      <span style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:11,color:"var(--color-text-tertiary)",width:36,textAlign:"right"}}>{pct.toFixed(1)}%</span>
    </div>
  );
}

export default function DQPage(){
  const [dq,setDq]=useState<DQSummaryResponse|null>(null);
  const [drift,setDrift]=useState<DriftResponse|null>(null);
  const [loading,setLoading]=useState(true);
  const [driftLoading,setDriftLoading]=useState(false);
  const [driftRef,setDriftRef]=useState("1-24");
  const [driftCur,setDriftCur]=useState("25-36");

  useEffect(()=>{
    dqApi.summary().then(r=>setDq(r.data)).catch(()=>{}).finally(()=>setLoading(false));
  },[]);

  const loadDrift=async()=>{
    setDriftLoading(true);
    try{ const r=await driftApi.detect({ref:driftRef,cur:driftCur}); setDrift(r.data); }
    catch{} finally{setDriftLoading(false);}
  };

  const total=dq?((dq.grade_distribution.A??0)+(dq.grade_distribution.B??0)+(dq.grade_distribution.C??0)+(dq.grade_distribution.D??0)+(dq.grade_distribution.F??0)):0;
  const meanScore = dq ? (dq.mean_dq ?? (dq.mean_score != null ? dq.mean_score * 100 : null)) : null;
  const medianScore = dq ? (dq.median_dq ?? (dq.median_score != null ? dq.median_score * 100 : null)) : null;
  const dims = dq ? (dq.mean_dimension_scores ?? dq.dimension_means) : undefined;

  return(
    <div className="animate-fade-up" style={{maxWidth:1200,margin:"0 auto",display:"flex",flexDirection:"column",gap:24}}>
      {/* DQ Summary */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:20}}>
        {/* Score + Grades */}
        <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:16}}><ShieldCheck size={16} color="var(--color-accent)"/><p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)"}}>DQ Score</p></div>
          {loading?<SkeletonCard rows={3}/>:(
            <>
              <div style={{display:"flex",gap:20,marginBottom:20}}>
                <div><p style={{fontSize:11,color:"var(--color-text-tertiary)",textTransform:"uppercase",letterSpacing:"0.06em",marginBottom:4}}>Mean Score</p><p style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:28,fontWeight:600,color:"var(--color-text-primary)"}}>{meanScore!=null?meanScore.toFixed(1):"--"}%</p></div>
                <div><p style={{fontSize:11,color:"var(--color-text-tertiary)",textTransform:"uppercase",letterSpacing:"0.06em",marginBottom:4}}>Median Score</p><p style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:28,fontWeight:600,color:"var(--color-text-primary)"}}>{medianScore!=null?medianScore.toFixed(1):"--"}%</p></div>
              </div>
              {dq&&<div style={{display:"flex",flexDirection:"column",gap:8}}>
                {(["A","B","C","D","F"] as const).map(g=><GradeBar key={g} grade={g} count={dq.grade_distribution[g]??0} total={total}/>)}
              </div>}
            </>
          )}
        </div>

        {/* Radar */}
        <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
          <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",marginBottom:12}}>DQ Dimensions</p>
          {loading?<SkeletonCard rows={4}/>:dims?<DQRadarChart dimensions={dims}/>:<p style={{fontSize:13,color:"var(--color-text-tertiary)"}}>No data</p>}
        </div>

        {/* Top Violations */}
        <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
          <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",marginBottom:12}}>Top Violated Rules</p>
          {loading?<SkeletonCard rows={5}/>:dq?.top_violated_rules&&dq.top_violated_rules.length>0?(
            <div style={{display:"flex",flexDirection:"column",gap:8}}>
              {dq.top_violated_rules.slice(0,8).map(r=>(
                <div key={r.rule_id} style={{display:"flex",alignItems:"center",gap:8,paddingBottom:8,borderBottom:"1px solid var(--color-border)"}}>
                  <XCircle size={13} color="#EF4444" style={{flexShrink:0}}/>
                  <div style={{flex:1,minWidth:0}}>
                    <p style={{fontSize:12,fontWeight:500,color:"var(--color-text-primary)",fontFamily:"var(--font-geist-mono),monospace"}}>{r.rule_id}</p>
                    <p style={{fontSize:11,color:"var(--color-text-tertiary)",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{r.description}</p>
                  </div>
                  <span style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:(r.violation_rate??0)>=.1?"#EF4444":"#F59E0B",flexShrink:0}}>{((r.violation_rate??0)*100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          ):<p style={{fontSize:13,color:"var(--color-text-tertiary)"}}>No violations data.</p>}
        </div>
      </div>

      {/* DQ Trend over the panel */}
      <div className="gs-widget" style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
        <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",marginBottom:4}}>DQ Trend by Month</p>
        <p style={{fontSize:12,color:"var(--color-text-tertiary)",marginBottom:12}}>Measured per-month, not modeled — read directly off {dq?.by_month?.length ?? 0} scored months.</p>
        {loading?<div className="skeleton skel-chart"/>:dq?.by_month&&dq.by_month.length>0?<DQTrendChart data={dq.by_month}/>:<p style={{fontSize:13,color:"var(--color-text-tertiary)"}}>No monthly series returned.</p>}
      </div>

      {/* Drift Section */}
      <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:16}}>
          <div style={{display:"flex",alignItems:"center",gap:8}}><ArrowsLeftRight size={16} color="var(--color-accent)"/><p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)"}}>Feature Drift Monitor</p></div>
          <div style={{display:"flex",gap:10,alignItems:"center"}}>
            <input value={driftRef} onChange={e=>setDriftRef(e.target.value)} placeholder="Reference: 1-24" style={{width:120,background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)",borderRadius:5,padding:"5px 8px",fontSize:12,color:"var(--color-text-primary)",fontFamily:"var(--font-geist-mono),monospace"}}/>
            <span style={{fontSize:12,color:"var(--color-text-tertiary)"}}>vs</span>
            <input value={driftCur} onChange={e=>setDriftCur(e.target.value)} placeholder="Current: 25-36" style={{width:120,background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)",borderRadius:5,padding:"5px 8px",fontSize:12,color:"var(--color-text-primary)",fontFamily:"var(--font-geist-mono),monospace"}}/>
            <button onClick={loadDrift} disabled={driftLoading} style={{padding:"6px 14px",borderRadius:5,background:"var(--color-accent)",border:"none",color:"#fff",fontSize:12,fontWeight:500,cursor:"pointer"}}>{driftLoading?"Analyzing...":"Detect Drift"}</button>
          </div>
        </div>
        {drift?(
          <>
            <div style={{display:"flex",gap:20,marginBottom:16}}>
              <div style={{padding:"10px 16px",background:drift.retraining_trigger?.retrain_required?"rgba(239,68,68,.08)":"rgba(34,197,94,.08)",border:`1px solid ${drift.retraining_trigger?.retrain_required?"rgba(239,68,68,.2)":"rgba(34,197,94,.2)"}`,borderRadius:6,display:"flex",alignItems:"center",gap:8}}>
                {drift.retraining_trigger?.retrain_required?<WarningCircle size={14} color="#EF4444"/>:<CheckCircle size={14} color="#22C55E"/>}
                <span style={{fontSize:13,fontWeight:500,color:drift.retraining_trigger?.retrain_required?"#EF4444":"#22C55E"}}>{drift.retraining_trigger?.retrain_required?"Retraining Triggered":"Model Stable"}</span>
              </div>
              <div style={{padding:"10px 16px",background:"var(--color-bg-elevated)",borderRadius:6}}>
                <span style={{fontSize:12,color:"var(--color-text-tertiary)"}}>Adversarial AUC: </span>
                <span style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:13,fontWeight:600,color:((drift.retraining_trigger?.adversarial_auc ?? drift.adversarial?.adversarial_auc)??0)>=.8?"#EF4444":"#22C55E"}}>{((drift.retraining_trigger?.adversarial_auc ?? drift.adversarial?.adversarial_auc)??0).toFixed(4)}</span>
              </div>
            </div>
            <div style={{overflowX:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:13}}>
                <thead><tr>{["Feature","PSI","KS Stat","JS Div","Missingness Delta","Verdict"].map(h=>(<th key={h} style={{fontSize:11,letterSpacing:"0.06em",textTransform:"uppercase",color:"var(--color-text-tertiary)",fontWeight:400,textAlign:"left",padding:"8px 10px",borderBottom:"1px solid var(--color-border)",whiteSpace:"nowrap"}}>{h}</th>))}</tr></thead>
                <tbody>{drift.features.slice(0,20).map(f=>(
                  <tr key={f.feature} style={{borderBottom:"1px solid var(--color-border)",transition:"background 100ms"}} onMouseEnter={e=>{(e.currentTarget as HTMLElement).style.background="var(--color-bg-elevated)";}} onMouseLeave={e=>{(e.currentTarget as HTMLElement).style.background="transparent";}}>
                    <td style={{padding:"8px 10px",fontSize:12,color:"var(--color-text-primary)"}}>{f.feature}</td>
                    <td style={{padding:"8px 10px",fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:(f.psi??0)>=.25?"#EF4444":(f.psi??0)>=.1?"#F59E0B":"#22C55E"}}>{(f.psi??0).toFixed(4)}</td>
                    <td style={{padding:"8px 10px",fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:"var(--color-text-secondary)"}}>{(f.ks_stat??0).toFixed(4)}</td>
                    <td style={{padding:"8px 10px",fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:"var(--color-text-secondary)"}}>{((f.js_div??f.js_divergence)??0).toFixed(4)}</td>
                    <td style={{padding:"8px 10px",fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:Math.abs(f.missingness_delta??f.missing_delta??0)>=.05?"#EF4444":"var(--color-text-secondary)"}}>{(f.missingness_delta??f.missing_delta??0)>=0?"+":""}{(f.missingness_delta??f.missing_delta??0).toFixed(4)}</td>
                    <td style={{padding:"8px 10px"}}><span style={{display:"inline-block",padding:"2px 7px",borderRadius:4,fontSize:10,fontFamily:"var(--font-geist-mono),monospace",fontWeight:600,background:f.verdict==="KEEP"?"rgba(34,197,94,.1)":f.verdict==="MONITOR"?"rgba(245,158,11,.1)":"rgba(239,68,68,.1)",color:f.verdict==="KEEP"?"#22C55E":f.verdict==="MONITOR"?"#F59E0B":"#EF4444"}}>{f.verdict}</span></td>
                  </tr>
                ))}
                </tbody>
              </table>
            </div>
          </>
        ):<div style={{padding:"32px",textAlign:"center",color:"var(--color-text-tertiary)",fontSize:14}}><ArrowsLeftRight size={28} style={{margin:"0 auto 10px",display:"block"}}/><p>Select reference and current month windows, then run drift detection.</p></div>}
      </div>
    </div>
  );
}
