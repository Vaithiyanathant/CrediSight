"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { predictionApi, survivalApi, explainApi, reviewerApi } from "@/lib/api";
import { RiskBadge } from "@/components/ui/RiskBadge";
import { ProbBar } from "@/components/ui/ProbBar";
import { AiBanner } from "@/components/ui/AiBanner";
import { SurvivalChart } from "@/components/charts/SurvivalChart";
import { ShapWaterfallChart } from "@/components/charts/ShapWaterfallChart";
import { SkeletonCard } from "@/components/ui/Skeleton";
import type { PredictionResponse, SurvivalResponse, LocalExplainResponse } from "@/types";
import { ArrowLeft, CheckCircle, XCircle, ArrowUpRight } from "@phosphor-icons/react";
import Link from "next/link";

function fmt(n:number,type:"pct"|"usd"|"num"="num"):string{
  if(type==="pct") return `${(n*100).toFixed(2)}%`;
  if(type==="usd"){ if(n>=1e6)return `$${(n/1e6).toFixed(1)}M`; if(n>=1e3)return `$${(n/1e3).toFixed(0)}K`; return `$${n.toFixed(0)}`; }
  return n.toLocaleString();
}

export default function LoanPage(){
  const { id } = useParams<{id:string}>();
  const [pred,setPred]=useState<PredictionResponse|null>(null);
  const [surv,setSurv]=useState<SurvivalResponse|null>(null);
  const [expl,setExpl]=useState<LocalExplainResponse|null>(null);
  const [loading,setLoading]=useState(true);
  const [decision,setDecision]=useState<"Confirm"|"Reject"|"Escalate"|null>(null);
  const [rationale,setRationale]=useState("");
  const [submitting,setSubmitting]=useState(false);
  const [submitted,setSubmitted]=useState(false);
  const [error,setError]=useState<string|null>(null);

  useEffect(()=>{
    if(!id) return;
    setLoading(true);
    Promise.all([
      predictionApi.byLoanId(id).catch(()=>null),
      survivalApi.byLoanId(id).catch(()=>null),
      explainApi.local(id,{head:"next_12m_default"}).catch(()=>null),
    ]).then(([p,s,e])=>{
      if(p) setPred(p.data);
      if(s) setSurv(s.data);
      if(e) setExpl(e.data);
    }).finally(()=>setLoading(false));
  },[id]);

  const submitDecision=async()=>{
    if(!decision||!pred) return;
    setSubmitting(true);
    try{
      await reviewerApi.decision({loan_id:id,month_index:pred.as_of_month??pred.month_index??0,human_decision:decision,model_recommendation:pred.reviewer_action,rationale,anomaly_score:pred.anomaly?.score??pred.anomaly_score});
      setSubmitted(true);
    }catch(e:unknown){ setError(e instanceof Error?e.message:"Failed to submit"); }
    finally{ setSubmitting(false); }
  };

  if(loading) return(
    <div style={{maxWidth:1200,margin:"0 auto"}}>
      <div style={{display:"grid",gridTemplateColumns:"2fr 1fr",gap:20}}>
        <div style={{display:"flex",flexDirection:"column",gap:20}}>
          <SkeletonCard rows={6}/><SkeletonCard rows={4}/>
        </div>
        <div style={{display:"flex",flexDirection:"column",gap:20}}>
          <SkeletonCard rows={5}/><SkeletonCard rows={3}/>
        </div>
      </div>
    </div>
  );

  const p=pred;
  const anomalyScore = p?.anomaly?.score ?? p?.anomaly_score ?? 0;
  const anomalyTier  = p?.anomaly?.tier  ?? p?.anomaly_tier  ?? "low";
  const dqGrade      = p?.data_quality?.dq_grade ?? p?.dq_grade ?? "—";
  const reviewerAction = p?.reviewer_action ?? "No Action";
  const topDrivers   = p?.explanation?.top_drivers ?? p?.top_drivers ?? [];
  const getPred = (key: string) => {
    const preds = p?.predictions ?? {};
    return (preds as Record<string,{value:number;lower?:number;upper?:number;ci?:[number,number]}>)[`prob_${key}`]
        ?? (preds as Record<string,{value:number;lower?:number;upper?:number}>)[key]
        ?? {value:0};
  };
  // `.probs` is the per-state map; the sibling keys on next_state are
  // `predicted` (a state name string) and `confidence`, which must not be
  // charted as probabilities. Guarded so a numeric value is always produced.
  const nextStateBlock = p?.predictions?.next_state;
  const nextState: Record<string, number> = nextStateBlock?.probs ?? {};
  const predictedState = nextStateBlock?.predicted;
  const risk=(v:number):"low"|"medium"|"high"=>v>=.3?"high":v>=.1?"medium":"low";

  return(
    <div className="animate-fade-up" style={{maxWidth:1200,margin:"0 auto"}}>
      <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:24}}>
        <Link href="/portfolio" style={{display:"inline-flex",alignItems:"center",gap:6,fontSize:13,color:"var(--color-text-secondary)",textDecoration:"none"}}><ArrowLeft size={14}/>Portfolio</Link>
        <span style={{color:"var(--color-border)"}}>/</span>
        <span style={{fontSize:13,fontFamily:"var(--font-geist-mono),monospace",color:"var(--color-text-primary)"}}>{id}</span>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"2fr 1fr",gap:20}}>
        {/* Left column */}
        <div style={{display:"flex",flexDirection:"column",gap:20}}>
          {/* Header card */}
          {p&&(
            <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
              <div style={{display:"flex",alignItems:"flex-start",justifyContent:"space-between",marginBottom:20}}>
                <div>
                  <p style={{fontSize:11,color:"var(--color-text-tertiary)",letterSpacing:"0.06em",textTransform:"uppercase",marginBottom:4}}>Loan ID</p>
                  <p style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:18,fontWeight:600,color:"var(--color-text-primary)"}}>{p.loan_id}</p>
                </div>
                <div style={{display:"flex",gap:8,alignItems:"center"}}>
                  <RiskBadge level={p.current_status==="Default"||p.current_status==="90DPD"?"high":p.current_status==="60DPD"||p.current_status==="30DPD"?"medium":"low"} label={p.current_status}/>
                  <RiskBadge level={anomalyTier.includes("high")?"high":anomalyTier.includes("medium")||anomalyTier.includes("WARNING")?"medium":"low"} label={anomalyTier}/>
                </div>
              </div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:16}}>
                <div>
                  <p style={{fontSize:11,color:"var(--color-text-tertiary)",letterSpacing:"0.06em",textTransform:"uppercase",marginBottom:6}}>Current Balance</p>
                  <p style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:22,fontWeight:600,color:"var(--color-text-primary)"}}>{fmt(p.current_balance,"usd")}</p>
                </div>
                <div>
                  <p style={{fontSize:11,color:"var(--color-text-tertiary)",letterSpacing:"0.06em",textTransform:"uppercase",marginBottom:6}}>Anomaly Score</p>
                  <p style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:22,fontWeight:600,color:anomalyScore>=.6?"#EF4444":anomalyScore>=.3?"#F59E0B":"#22C55E"}}>{anomalyScore.toFixed(3)}</p>
                </div>
                <div>
                  <p style={{fontSize:11,color:"var(--color-text-tertiary)",letterSpacing:"0.06em",textTransform:"uppercase",marginBottom:6}}>DQ Grade</p>
                  <p style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:22,fontWeight:600,color:dqGrade==="A"?"#22C55E":dqGrade==="B"?"#84CC16":dqGrade==="C"?"#EAB308":"#EF4444"}}>{dqGrade}</p>
                </div>
              </div>
            </div>
          )}

          {/* Predictions */}
          {p&&(
            <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
              <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",marginBottom:16}}>Risk Predictions</p>
              <div style={{display:"flex",flexDirection:"column",gap:16}}>
                {([
                  {label:"12m Default Probability",  v:getPred("next_12m_default")},
                  {label:"3m Delinquency Probability",v:getPred("next_3m_delinquency")},
                  {label:"6m Delinquency Probability",v:getPred("next_6m_delinquency")},
                  {label:"12m Prepayment Probability",v:getPred("next_12m_prepayment")},
                ] as {label:string;v:{value:number;lower?:number;upper?:number}}[]).map(({label,v})=>(
                  <div key={label}>
                    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
                      <p style={{fontSize:13,color:"var(--color-text-secondary)"}}>{label}</p>
                      <RiskBadge level={risk(v.value)} label={`${(v.value*100).toFixed(1)}%`} size="sm"/>
                    </div>
                    <ProbBar value={v.value} lower={v.lower} upper={v.upper}/>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Survival */}
          {surv&&(
            <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
              <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",marginBottom:4}}>Survival & CIF Curves</p>
              <p style={{fontSize:12,color:"var(--color-text-tertiary)",marginBottom:16}}>Kaplan-Meier survival with competing risk CIF for default and prepayment</p>
              <SurvivalChart curve={surv.curve ?? surv.horizons_m?.map((t,i)=>({t, survival: surv.survival?.[i]??0, cif_default: surv.cif_default?.[i]??0, cif_prepay: surv.cif_prepay?.[i]??0})) ?? []}/>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,marginTop:12}}>
                <div style={{background:"var(--color-bg-elevated)",borderRadius:6,padding:"10px 14px"}}>
                  <p style={{fontSize:11,color:"var(--color-text-tertiary)",textTransform:"uppercase",letterSpacing:"0.06em"}}>12m Default Prob</p>
                  <p style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:18,fontWeight:600,color:"#EF4444",marginTop:2}}>{(surv.prob_default_12m??0).toFixed(4)}</p>
                </div>
                <div style={{background:"var(--color-bg-elevated)",borderRadius:6,padding:"10px 14px"}}>
                  <p style={{fontSize:11,color:"var(--color-text-tertiary)",textTransform:"uppercase",letterSpacing:"0.06em"}}>12m Prepay Prob</p>
                  <p style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:18,fontWeight:600,color:"#6366F1",marginTop:2}}>{(surv.prob_prepay_12m??0).toFixed(4)}</p>
                </div>
              </div>
            </div>
          )}

          {/* SHAP */}
          {expl&&p&&(
            <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
              <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",marginBottom:4}}>SHAP Feature Attribution</p>
              <p style={{fontSize:12,color:"var(--color-text-tertiary)",marginBottom:16}}>Top feature contributions to 12m default probability</p>
              <ShapWaterfallChart drivers={(expl.shap_values??expl.top_contributions??[]).map(s=>({feature:s.feature,shap_value:s.shap_value??s.shap??0,direction:(s.shap_value??s.shap??0)>0?"positive":"negative" as "positive"|"negative",rank:s.rank??0}))} baseValue={expl.base_value} prediction={expl.prediction??expl.probability??0}/>
            </div>
          )}
        </div>

        {/* Right column */}
        <div style={{display:"flex",flexDirection:"column",gap:20}}>
          {/* Next state */}
          {p&&(
            <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
              <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:16}}>
                <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",margin:0}}>Next State Distribution</p>
                {predictedState&&(
                  <span style={{marginLeft:"auto",fontSize:11,fontWeight:600,padding:"2px 9px",borderRadius:999,background:"var(--color-bg-subtle)",color:"var(--color-text-secondary)",fontFamily:"var(--font-geist-mono),monospace"}}>
                    predicted: {predictedState}
                  </span>
                )}
              </div>
              <div style={{display:"flex",flexDirection:"column",gap:10}}>
                {Object.entries(nextState)
                  .map(([state,raw])=>[state, typeof raw==="number"&&Number.isFinite(raw)?raw:0] as const)
                  .sort((a,b)=>b[1]-a[1])
                  .map(([state,prob])=>(
                  <div key={state}>
                    <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
                      <span style={{fontSize:12,color:state===predictedState?"var(--color-text-primary)":"var(--color-text-secondary)",fontWeight:state===predictedState?600:400}}>{state}</span>
                      <span style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:"var(--color-text-primary)"}}>{prob.toFixed(4)}</span>
                    </div>
                    <div style={{height:4,background:"var(--color-bg-subtle)",borderRadius:2,overflow:"hidden"}}>
                      <div style={{height:"100%",width:`${Math.min(prob*100,100)}%`,background:state==="Default"||state==="90DPD"?"var(--color-risk-high)":state==="60DPD"||state==="30DPD"?"var(--color-risk-medium)":state==="Prepaid"?"var(--color-survival)":state==="Closed"?"var(--chart-5)":"var(--color-risk-low)",borderRadius:2,transition:"width .5s var(--ease-out)"}}/>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Reviewer Card */}
          {p&&(
            <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
              <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",marginBottom:4}}>Reviewer Decision</p>
              <p style={{fontSize:12,color:"var(--color-text-tertiary)",marginBottom:12}}>Model recommends: <span style={{color:reviewerAction==="Escalate"?"#EF4444":reviewerAction==="Flag"?"#F59E0B":"#A1A1AA",fontWeight:500}}>{reviewerAction}</span></p>
              {submitted?(
                <div style={{background:"rgba(34,197,94,.08)",border:"1px solid rgba(34,197,94,.2)",borderRadius:6,padding:"10px 12px",display:"flex",alignItems:"center",gap:8}}>
                  <CheckCircle size={16} color="#22C55E"/>
                  <span style={{fontSize:13,color:"#22C55E"}}>Decision recorded: {decision}</span>
                </div>
              ):(
                <>
                  <div style={{display:"flex",gap:8,marginBottom:12}}>
                    {(["Confirm","Reject","Escalate"] as const).map(d=>(
                      <button key={d} onClick={()=>setDecision(d)} style={{flex:1,padding:"8px 0",borderRadius:6,fontSize:12,fontWeight:500,cursor:"pointer",border:`1px solid ${decision===d?d==="Confirm"?"#22C55E":d==="Reject"?"#EF4444":"#F59E0B":"var(--color-border)"}`,background:decision===d?d==="Confirm"?"rgba(34,197,94,.1)":d==="Reject"?"rgba(239,68,68,.1)":"rgba(245,158,11,.1)":"transparent",color:decision===d?d==="Confirm"?"#22C55E":d==="Reject"?"#EF4444":"#F59E0B":"var(--color-text-secondary)",transition:"all 150ms ease"}}>{d}</button>
                    ))}
                  </div>
                  <textarea value={rationale} onChange={e=>setRationale(e.target.value)} placeholder="Rationale (optional)" rows={3} style={{width:"100%",background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)",borderRadius:6,padding:"8px 10px",fontSize:12,color:"var(--color-text-primary)",fontFamily:"var(--font-geist-sans),sans-serif",resize:"vertical",marginBottom:10}}/>
                  {error&&<p style={{fontSize:12,color:"#EF4444",marginBottom:8}}>{error}</p>}
                  <button onClick={submitDecision} disabled={!decision||submitting} style={{width:"100%",padding:"9px 0",borderRadius:6,fontSize:13,fontWeight:500,cursor:decision?"pointer":"not-allowed",border:"none",background:decision?"var(--color-accent)":"var(--color-bg-subtle)",color:decision?"#fff":"var(--color-text-tertiary)",transition:"all 150ms ease"}}>
                    {submitting?"Submitting...":"Submit Decision"}
                  </button>
                </>
              )}
            </div>
          )}

          {/* Top drivers */}
          {p&&topDrivers.length>0&&(
            <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
              <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",marginBottom:14}}>Top Risk Drivers</p>
              <div style={{display:"flex",flexDirection:"column",gap:10}}>
                {topDrivers.slice(0,8).map((d,i)=>{
                  const shapVal = (d as {shap_value?:number;shap?:number}).shap_value ?? (d as {shap?:number}).shap ?? 0;
                  const feat = d.feature;
                  return (
                  <div key={feat} style={{display:"flex",alignItems:"center",gap:10}}>
                    <span style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:11,color:"var(--color-text-tertiary)",width:16}}>#{i+1}</span>
                    <span style={{flex:1,fontSize:12,color:"var(--color-text-secondary)",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{feat}</span>
                    <span style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:shapVal>0?"#EF4444":"#22C55E",flexShrink:0}}>{shapVal>0?"+":""}{shapVal.toFixed(4)}</span>
                  </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
