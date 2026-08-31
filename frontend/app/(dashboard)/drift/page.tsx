"use client";
import { useState, useEffect, useCallback, useRef } from "react";
import { driftApi } from "@/lib/api";
import { KpiCard } from "@/components/ui/KpiCard";
import { SkeletonCard } from "@/components/ui/Skeleton";
import type { DriftResponse, DriftFeature } from "@/types";
import { ArrowsLeftRight, WarningCircle, CheckCircle, XCircle, ArrowDown, ArrowUp } from "@phosphor-icons/react";

const VERDICT_CFG: Record<string,{color:string;bg:string}> = {
  KEEP:              { color:"#22C55E", bg:"rgba(34,197,94,.08)"  },
  MONITOR:           { color:"#F59E0B", bg:"rgba(245,158,11,.08)" },
  DROP_OR_ROBUSTIFY: { color:"#EF4444", bg:"rgba(239,68,68,.08)"  },
};

function VerdictBadge({ verdict }: { verdict: string }) {
  const cfg = VERDICT_CFG[verdict] ?? { color:"#A1A1AA", bg:"rgba(161,161,170,.08)" };
  const icon = verdict==="KEEP" ? <CheckCircle size={12} color={cfg.color}/>
    : verdict==="MONITOR" ? <WarningCircle size={12} color={cfg.color}/>
    : <XCircle size={12} color={cfg.color}/>;
  return (
    <span style={{display:"inline-flex",alignItems:"center",gap:5,background:cfg.bg,color:cfg.color,borderRadius:4,padding:"2px 8px",fontSize:11,fontWeight:600,fontFamily:"var(--font-geist-mono),monospace"}}>
      {icon}{verdict.replace("_OR_"," / ")}
    </span>
  );
}

function MetricBar({ value, max, color }: { value:number; max:number; color:string }) {
  const pct = max > 0 ? Math.min((value/max)*100,100) : 0;
  return (
    <div style={{flex:1,height:6,background:"var(--color-bg-subtle)",borderRadius:3,overflow:"hidden"}}>
      <div style={{height:"100%",width:`${pct}%`,background:color,borderRadius:3,transition:"width .4s cubic-bezier(0.16,1,0.3,1)"}}/>
    </div>
  );
}

export default function DriftPage() {
  const [ref, setRef]       = useState("1-24");
  const [cur, setCur]       = useState("25-36");
  const [topK, setTopK]     = useState(20);
  const [data, setData]     = useState<DriftResponse|null>(null);
  const [loading,setLoading]= useState(false);
  const [error,setError]    = useState<string|null>(null);
  const [sortKey,setSortKey]= useState<keyof DriftFeature>("psi");
  const [sortDir,setSortDir]= useState<"asc"|"desc">("desc");
  const [filter,setFilter]  = useState("all");

  // Runs the default window comparison on mount rather than leaving the page
  // blank until the user finds the "Run Detection" button — drift status is
  // the whole point of the screen, and the default 1-24 vs 25-36 windows are
  // the ones the engine itself documents. `data` is kept on screen while a
  // new window is fetched so the page never flashes back to empty.
  const run = useCallback(async () => {
    setLoading(true); setError(null);
    try { const r = await driftApi.detect({ref,cur}); setData(r.data); }
    catch (e:unknown) { setError(e instanceof Error ? e.message : "Drift detection failed"); setData(null); }
    finally { setLoading(false); }
  }, [ref, cur]);

  const didAutoRun = useRef(false);
  useEffect(() => {
    // Mount-only: `run` changes identity whenever the window inputs change,
    // and re-firing a 3s request on every keystroke there would be worse than
    // useless. Subsequent runs are explicit, via the button.
    if (didAutoRun.current) return;
    didAutoRun.current = true;
    run();
  }, [run]);

  const handleSort = (k:keyof DriftFeature) => {
    if (k===sortKey) setSortDir(d=>d==="asc"?"desc":"asc");
    else { setSortKey(k); setSortDir("desc"); }
  };

  const all = data?.features ?? [];
  const rows = [...all]
    .filter(f=>filter==="all"||f.verdict===filter)
    .sort((a,b)=>{ const av=a[sortKey] as number|string,bv=b[sortKey] as number|string; const c=av<bv?-1:av>bv?1:0; return sortDir==="desc"?-c:c; })
    .slice(0, topK);

  const maxPsi = Math.max(...all.map(f=>f.psi??0),0.01);
  const maxKs  = Math.max(...all.map(f=>f.ks_stat??0),0.01);
  const cnt = (v:string) => all.filter(f=>f.verdict===v).length;

  const TH = ({col,label}:{col:keyof DriftFeature;label:string}) => (
    <th onClick={()=>handleSort(col)} style={{fontSize:11,letterSpacing:"0.06em",textTransform:"uppercase",color:sortKey===col?"var(--color-accent)":"var(--color-text-tertiary)",fontWeight:400,textAlign:"left",padding:"10px 12px",borderBottom:"1px solid var(--color-border)",cursor:"pointer",whiteSpace:"nowrap",userSelect:"none"}}>
      <span style={{display:"inline-flex",alignItems:"center",gap:4}}>{label}{sortKey===col?(sortDir==="desc"?<ArrowDown size={10}/>:<ArrowUp size={10}/>):null}</span>
    </th>
  );

  return (
    <div className="animate-fade-up" style={{maxWidth:1200,margin:"0 auto",display:"flex",flexDirection:"column",gap:24}}>

      {/* Controls */}
      <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
        <div style={{display:"flex",alignItems:"center",gap:12,flexWrap:"wrap"}}>
          <ArrowsLeftRight size={16} color="var(--color-accent)"/>
          <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",marginRight:8}}>Drift Detection</p>
          <div style={{display:"flex",alignItems:"center",gap:8}}>
            <label style={{fontSize:12,color:"var(--color-text-tertiary)"}}>Reference months:</label>
            <input value={ref} onChange={e=>setRef(e.target.value)} placeholder="e.g. 1-24" style={{width:90,background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)",borderRadius:6,padding:"6px 10px",fontSize:13,color:"var(--color-text-primary)",fontFamily:"var(--font-geist-mono),monospace"}}/>
          </div>
          <div style={{display:"flex",alignItems:"center",gap:8}}>
            <label style={{fontSize:12,color:"var(--color-text-tertiary)"}}>Current months:</label>
            <input value={cur} onChange={e=>setCur(e.target.value)} placeholder="e.g. 25-36" style={{width:90,background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)",borderRadius:6,padding:"6px 10px",fontSize:13,color:"var(--color-text-primary)",fontFamily:"var(--font-geist-mono),monospace"}}/>
          </div>
          <div style={{display:"flex",alignItems:"center",gap:8}}>
            <label style={{fontSize:12,color:"var(--color-text-tertiary)"}}>Top K:</label>
            <input type="number" value={topK} onChange={e=>setTopK(Number(e.target.value))} min={5} max={145} style={{width:65,background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)",borderRadius:6,padding:"6px 10px",fontSize:13,color:"var(--color-text-primary)",fontFamily:"var(--font-geist-mono),monospace"}}/>
          </div>
          <button onClick={run} disabled={loading} style={{marginLeft:"auto",padding:"8px 20px",borderRadius:6,fontSize:13,fontWeight:500,background:"var(--color-accent)",border:"none",color:"#fff",cursor:loading?"not-allowed":"pointer",opacity:loading?0.7:1,transition:"all 150ms ease"}}>
            {loading?"Detecting…":"Run Detection"}
          </button>
        </div>
        {error&&<p style={{fontSize:13,color:"#EF4444",marginTop:10}}>{error}</p>}
      </div>

      {/* KPI summary */}
      {data&&(
        <div className="gs-panel">
          <div className="gs-panel__head">
            <p className="gs-panel__title">Drift overview</p>
            <span style={{fontSize:11,color:"var(--color-text-tertiary)",fontFamily:"var(--font-geist-mono),monospace"}}>
              {data.ref_window} vs {data.cur_window} · {(data.elapsed_ms??0).toFixed(0)}ms
            </span>
          </div>
          <div data-kpi-row style={{gridTemplateColumns:"repeat(4,1fr)",gap:14}}>
            <KpiCard label="Features compared" icon={ArrowsLeftRight} tone="running" value={all.length.toString()}
              sub={`${(data.n_reference_rows??0).toLocaleString()} vs ${(data.n_current_rows??0).toLocaleString()} rows`}/>
            <KpiCard label="Stable (keep)" icon={CheckCircle} tone="completed" value={cnt("KEEP").toString()}
              sub="within PSI threshold"/>
            <KpiCard label="Monitor" icon={WarningCircle} tone="wait" value={cnt("MONITOR").toString()}
              sub="PSI 0.10 – 0.25"/>
            <KpiCard label="Drop / robustify" icon={XCircle} tone={cnt("DROP_OR_ROBUSTIFY")>0?"failed":"completed"}
              value={cnt("DROP_OR_ROBUSTIFY").toString()}
              chip={maxPsi>0?`max PSI ${maxPsi.toFixed(2)}`:undefined} chipIcon={WarningCircle}/>
          </div>
        </div>
      )}

      {/* Retraining verdict — states the decision, the evidence behind it, and
          what adversarial AUC actually measures, rather than repeating the
          same sentence twice with no definition. */}
      {data&&(()=>{
        const trig=data.retraining_trigger;
        const adv=trig?.adversarial_auc ?? data.adversarial?.adversarial_auc ?? 0;
        // Threshold comes from the backend, never a hardcoded UI constant —
        // this panel previously compared against 0.6 while the engine decided
        // at 0.8, so the banner could contradict its own verdict.
        const advThreshold=trig?.thresholds?.adversarial_auc ?? 0.8;
        const required=!!trig?.retrain_required;
        const tone=required
          ? {fg:"var(--color-risk-high)",bg:"hsl(354 80% 65% / .09)",bd:"hsl(354 80% 65% / .28)"}
          : {fg:"var(--color-risk-low)", bg:"hsl(142 65% 52% / .10)", bd:"hsl(142 65% 52% / .28)"};
        return (
          <div style={{background:tone.bg,border:`1px solid ${tone.bd}`,borderRadius:10,padding:"16px 20px"}}>
            <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:10}}>
              {required?<WarningCircle size={18} color={tone.fg} weight="fill"/>:<CheckCircle size={18} color={tone.fg} weight="fill"/>}
              <p style={{fontSize:14,fontWeight:600,color:tone.fg,margin:0}}>
                {required?"Retraining recommended":"Model stable — no retraining needed"}
              </p>
              <span style={{marginLeft:"auto",fontSize:11,fontWeight:600,padding:"2px 9px",borderRadius:999,
                background:"var(--color-bg-subtle)",color:"var(--color-text-secondary)",fontFamily:"var(--font-geist-mono),monospace"}}>
                batch: {data.batch_verdict}
              </span>
            </div>

            {/* The engine's own reasons, verbatim — this is the evidence the
                decision was actually made on. */}
            {trig?.reasons?.length>0&&(
              <ul style={{margin:"0 0 12px",padding:0,listStyle:"none",display:"flex",flexDirection:"column",gap:5}}>
                {trig.reasons.map(r=>(
                  <li key={r} style={{fontSize:12.5,color:"var(--color-text-secondary)",display:"flex",gap:8,alignItems:"flex-start"}}>
                    <span style={{color:tone.fg,lineHeight:1.5}}>•</span>
                    <span style={{fontFamily:"var(--font-geist-mono),monospace"}}>{r}</span>
                  </li>
                ))}
              </ul>
            )}

            {trig?.features_over_psi_threshold?.length>0&&(
              <div style={{display:"flex",flexWrap:"wrap",gap:6,marginBottom:12}}>
                <span style={{fontSize:11,color:"var(--color-text-tertiary)",marginRight:2}}>Shifted features:</span>
                {trig.features_over_psi_threshold.map(f=>(
                  <span key={f} style={{fontSize:11,padding:"2px 8px",borderRadius:4,background:"var(--color-bg-subtle)",
                    color:"var(--color-text-secondary)",fontFamily:"var(--font-geist-mono),monospace"}}>{f}</span>
                ))}
              </div>
            )}

            {/* Seasoning holdouts, stated rather than hidden. These features
                cross the PSI threshold on a perfectly healthy book — a reader
                who sees them missing from the list above deserves to know why
                the engine set them aside instead of wondering if it missed them. */}
            {(trig?.notes?.length??0)>0&&(
              <div style={{marginBottom:12,padding:"9px 11px",borderRadius:6,
                background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)"}}>
                {trig.notes!.map(n=>(
                  <p key={n} style={{fontSize:11.5,color:"var(--color-text-tertiary)",margin:0,lineHeight:1.55}}>
                    <span style={{color:"var(--color-text-secondary)",fontWeight:600}}>Excluded as seasoning · </span>{n}
                  </p>
                ))}
              </div>
            )}

            {/* What the number means, in one plain sentence — the previous
                copy quoted the AUC and then said "significant shift" without
                ever saying what was being measured. */}
            <div style={{borderTop:`1px solid ${tone.bd}`,paddingTop:11}}>
              <div style={{display:"flex",alignItems:"baseline",gap:8,flexWrap:"wrap",marginBottom:5}}>
                <span style={{fontSize:12,color:"var(--color-text-secondary)"}}>Adversarial validation AUC</span>
                <span style={{fontSize:15,fontWeight:600,color:adv>=advThreshold?tone.fg:"var(--color-text-primary)",fontFamily:"var(--font-geist-mono),monospace"}}>
                  {adv.toFixed(4)}
                </span>
                <span style={{fontSize:11,color:"var(--color-text-tertiary)",fontFamily:"var(--font-geist-mono),monospace"}}>
                  threshold {advThreshold} · 0.5 = indistinguishable
                </span>
              </div>
              <p style={{fontSize:12,color:"var(--color-text-tertiary)",margin:0,lineHeight:1.55}}>
                A throwaway classifier is trained to guess whether a row came from the reference window or
                the current one. At <strong style={{color:"var(--color-text-secondary)",fontWeight:600}}>0.5</strong> it
                cannot tell them apart, so the two periods look like the same population.
                At <strong style={{color:"var(--color-text-secondary)",fontWeight:600}}>{adv.toFixed(2)}</strong> it
                separates them {adv>=advThreshold?"easily":"only weakly"} — the windows differ in ways a model
                trained on the older one may not carry over
                {data.adversarial?.interpretation?<> ({data.adversarial.interpretation})</>:null}.
              </p>
              {/* Without naming the holdouts this number is unfalsifiable: the
                  forward-looking labels are null in the scoring window by
                  construction and would pin the AUC at 1.0 on their own. */}
              {(data.adversarial?.excluded_columns?.length||data.adversarial?.excluded_seasoning_columns?.length)?(
                <p style={{fontSize:11,color:"var(--color-text-tertiary)",margin:"7px 0 0",lineHeight:1.5}}>
                  Held out of the classifier:{" "}
                  <span style={{fontFamily:"var(--font-geist-mono),monospace"}}>
                    {[...(data.adversarial.excluded_columns||[]),
                      ...(data.adversarial.excluded_seasoning_columns||[])].join(", ")}
                  </span>{" "}
                  — identity, targets and calendar-driven columns separate the windows
                  perfectly by construction and would measure the clock, not the population.
                </p>
              ):null}
            </div>
          </div>
        );
      })()}

      {/* Feature table */}
      {loading&&<SkeletonCard rows={8}/>}
      {data&&!loading&&(
        <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8}}>
          <div style={{padding:"14px 20px",borderBottom:"1px solid var(--color-border)",display:"flex",alignItems:"center",gap:12}}>
            <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",flex:1}}>Feature Drift Report</p>
            <select value={filter} onChange={e=>setFilter(e.target.value)} style={{background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)",borderRadius:5,padding:"5px 8px",fontSize:12,color:"var(--color-text-secondary)",cursor:"pointer"}}>
              <option value="all">All verdicts</option>
              <option value="KEEP">KEEP</option>
              <option value="MONITOR">MONITOR</option>
              <option value="DROP_OR_ROBUSTIFY">DROP / ROBUSTIFY</option>
            </select>
            <span style={{fontSize:12,color:"var(--color-text-tertiary)",fontFamily:"var(--font-geist-mono),monospace"}}>{rows.length} features</span>
          </div>
          <div style={{overflowX:"auto"}}>
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:13}}>
              <thead><tr>
                <TH col="feature" label="Feature"/>
                <TH col="psi" label="PSI"/>
                <TH col="ks_stat" label="KS Stat"/>
                <TH col="js_divergence" label="JS Div"/>
                <TH col="missingness_delta" label={"Δ Missing"}/>
                <TH col="verdict" label="Verdict"/>
              </tr></thead>
              <tbody>
                {rows.map(f=>(
                  <tr key={f.feature} style={{borderBottom:"1px solid var(--color-border)",transition:"background 100ms"}} onMouseEnter={e=>{(e.currentTarget as HTMLElement).style.background="var(--color-bg-elevated)";}} onMouseLeave={e=>{(e.currentTarget as HTMLElement).style.background="transparent";}}>
                    <td style={{padding:"10px 12px",fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:"var(--color-text-primary)",maxWidth:200,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>
                      {f.feature}
                      {/* A DROP_OR_ROBUSTIFY verdict on a seasoning feature is
                          expected, not a finding — mark it so the row is not
                          read as a reason to retrain. */}
                      {f.seasoning&&(
                        <span title="Shifts deterministically with loan age — excluded from the retrain trigger"
                          style={{marginLeft:7,fontSize:9.5,fontWeight:600,letterSpacing:.4,padding:"1px 5px",borderRadius:3,
                            background:"var(--color-bg-subtle)",color:"var(--color-text-tertiary)",
                            border:"1px solid var(--color-border)",verticalAlign:"middle"}}>SEASONING</span>
                      )}
                    </td>
                    <td style={{padding:"10px 12px",minWidth:150}}>
                      <div style={{display:"flex",alignItems:"center",gap:8}}>
                        <MetricBar value={f.psi??0} max={maxPsi} color={(f.psi??0)>.2?"#EF4444":(f.psi??0)>.1?"#F59E0B":"#22C55E"}/>
                        <span style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:(f.psi??0)>.2?"#EF4444":(f.psi??0)>.1?"#F59E0B":"#22C55E",width:50,textAlign:"right"}}>{(f.psi??0).toFixed(4)}</span>
                      </div>
                    </td>
                    <td style={{padding:"10px 12px",minWidth:150}}>
                      <div style={{display:"flex",alignItems:"center",gap:8}}>
                        <MetricBar value={f.ks_stat??0} max={maxKs} color="#6366F1"/>
                        <span style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:"var(--color-text-secondary)",width:50,textAlign:"right"}}>{(f.ks_stat??0).toFixed(4)}</span>
                      </div>
                    </td>
                    <td style={{padding:"10px 12px",fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:"var(--color-text-secondary)",textAlign:"right"}}>{((f.js_div??f.js_divergence)??0).toFixed(4)}</td>
                    <td style={{padding:"10px 12px",fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:Math.abs(f.missingness_delta??f.missing_delta??0)>.05?"#F59E0B":"var(--color-text-secondary)",textAlign:"right"}}>{(f.missingness_delta??f.missing_delta??0)>0?"+":""}{((f.missingness_delta??f.missing_delta??0)*100).toFixed(2)}%</td>
                    <td style={{padding:"10px 12px"}}><VerdictBadge verdict={f.verdict}/></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!data&&!loading&&(
        <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"60px 40px",textAlign:"center"}}>
          <ArrowsLeftRight size={36} color="var(--color-text-tertiary)" style={{margin:"0 auto 16px"}}/>
          <p style={{fontSize:15,fontWeight:500,color:"var(--color-text-primary)",marginBottom:6}}>Configure and run drift detection</p>
          <p style={{fontSize:13,color:"var(--color-text-tertiary)"}}>Compare reference vs. current month windows using PSI, KS, and JS divergence</p>
        </div>
      )}
    </div>
  );
}
