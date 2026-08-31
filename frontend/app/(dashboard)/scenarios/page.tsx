"use client";
import { useEffect, useState } from "react";
import { scenarioApi } from "@/lib/api";
import { MonteCarloChart } from "@/components/charts/MonteCarloChart";
import { SkeletonCard } from "@/components/ui/Skeleton";
import type { ScenarioInfo, MonteCarloPath, ScenarioResponse } from "@/types";
import { Flask, Play, ChartLine } from "@phosphor-icons/react";

const SCENARIO_COLORS:Record<string,string>={"base":"#22C55E","Base":"#22C55E","adverse":"#EF4444","Adverse-Credit":"#EF4444","stagflation":"#F59E0B","Stagflation":"#F59E0B","high_prepayment":"#06B6D4","High-Prepayment":"#06B6D4","custom":"#6366F1"};

function getColor(name:string):string{ const k=Object.keys(SCENARIO_COLORS).find(k=>name.toLowerCase().includes(k)); return k?SCENARIO_COLORS[k]:"#6366F1"; }

export default function ScenariosPage(){
  const [scenarios,setScenarios]=useState<ScenarioInfo[]>([]);
  const [selected,setSelected]=useState<string|null>(null);
  const [result,setResult]=useState<ScenarioResponse|null>(null);
  const [loading,setLoading]=useState(true);
  const [running,setRunning]=useState(false);
  const [error,setError]=useState<string|null>(null);
  const [horizon,setHorizon]=useState(24);
  const [nPaths,setNPaths]=useState(1000);

  useEffect(()=>{
    scenarioApi.list().then(r=>{ setScenarios(r.data.scenarios??[]); if(r.data.scenarios?.length) setSelected(r.data.scenarios[0].scenario_name); }).catch(()=>{}).finally(()=>setLoading(false));
  },[]);

  const run=async()=>{
    if(!selected) return;
    setRunning(true); setError(null); setResult(null);
    try{
      const r=await scenarioApi.run({scenario:selected,n_paths:nPaths,horizon});
      setResult(r.data);
    }catch(e:unknown){ setError(e instanceof Error?e.message:"Failed to run scenario"); }
    finally{ setRunning(false); }
  };

  const sel=scenarios.find(s=>s.scenario_name===selected);

  return(
    <div className="animate-fade-up" style={{maxWidth:1200,margin:"0 auto"}}>
      <div style={{display:"grid",gridTemplateColumns:"320px 1fr",gap:20,alignItems:"start"}}>
        {/* Left: scenario selector */}
        <div style={{display:"flex",flexDirection:"column",gap:16}}>
          <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px"}}>
            <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",marginBottom:16,display:"flex",alignItems:"center",gap:8}}><Flask size={16} color="var(--color-accent)"/>Scenarios</p>
            {loading?<SkeletonCard rows={4}/>:(
              <div style={{display:"flex",flexDirection:"column",gap:6}}>
                {scenarios.map(s=>(
                  <button key={s.scenario_name} onClick={()=>setSelected(s.scenario_name)} style={{textAlign:"left",padding:"10px 12px",borderRadius:6,border:`1px solid ${selected===s.scenario_name?getColor(s.scenario_name)+"60":"var(--color-border)"}`,background:selected===s.scenario_name?getColor(s.scenario_name)+"18":"transparent",cursor:"pointer",transition:"all 150ms ease"}}>
                    <div style={{display:"flex",alignItems:"center",gap:8}}>
                      <div style={{width:8,height:8,borderRadius:"50%",background:getColor(s.scenario_name),flexShrink:0}}/>
                      <span style={{fontSize:13,fontWeight:selected===s.scenario_name?500:400,color:selected===s.scenario_name?"var(--color-text-primary)":"var(--color-text-secondary)"}}>{s.scenario_name}</span>
                    </div>
                    {selected===s.scenario_name&&s.description&&<p style={{fontSize:11,color:"var(--color-text-tertiary)",marginTop:6,lineHeight:1.4}}>{s.description}</p>}
                  </button>
                ))}
              </div>
            )}
          </div>

          {sel&&(
            <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px"}}>
              <p style={{fontSize:13,fontWeight:500,color:"var(--color-text-primary)",marginBottom:12}}>Macro Assumptions</p>
              <div style={{display:"flex",flexDirection:"column",gap:8}}>
                {[
                  {label:"GDP Growth",value:`${sel.gdp_growth_pct>0?"+":""}${sel.gdp_growth_pct}%`},
                  {label:"Unemployment",value:`${sel.unemployment_rate_pct}%`},
                  {label:"HPI Change",value:`${sel.hpi_change_pct>0?"+":""}${sel.hpi_change_pct}%`},
                  {label:"Rate Shock",value:`${sel.interest_rate_shock_bps>0?"+":""}${sel.interest_rate_shock_bps}bps`},
                  {label:"Default Mult.",value:`${sel.default_rate_multiplier}x`},
                ].map(({label,value})=>(
                  <div key={label} style={{display:"flex",justifyContent:"space-between",borderBottom:"1px solid var(--color-border)",paddingBottom:6}}>
                    <span style={{fontSize:12,color:"var(--color-text-tertiary)"}}>{label}</span>
                    <span style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:"var(--color-text-primary)"}}>{value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px"}}>
            <p style={{fontSize:13,fontWeight:500,color:"var(--color-text-primary)",marginBottom:12}}>Run Parameters</p>
            <div style={{display:"flex",flexDirection:"column",gap:10}}>
              <div>
                <label style={{fontSize:11,color:"var(--color-text-tertiary)",textTransform:"uppercase",letterSpacing:"0.06em",display:"block",marginBottom:4}}>Horizon (months)</label>
                <input type="number" value={horizon} onChange={e=>setHorizon(Number(e.target.value))} min={6} max={60} style={{width:"100%",background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)",borderRadius:6,padding:"7px 10px",fontSize:13,color:"var(--color-text-primary)",fontFamily:"var(--font-geist-mono),monospace"}}/>
              </div>
              <div>
                <label style={{fontSize:11,color:"var(--color-text-tertiary)",textTransform:"uppercase",letterSpacing:"0.06em",display:"block",marginBottom:4}}>Monte Carlo Paths</label>
                <input type="number" value={nPaths} onChange={e=>setNPaths(Number(e.target.value))} min={100} max={5000} step={100} style={{width:"100%",background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)",borderRadius:6,padding:"7px 10px",fontSize:13,color:"var(--color-text-primary)",fontFamily:"var(--font-geist-mono),monospace"}}/>
              </div>
              <button onClick={run} disabled={!selected||running} style={{width:"100%",padding:"10px 0",borderRadius:6,fontSize:13,fontWeight:500,cursor:selected?"pointer":"not-allowed",border:"none",background:"var(--color-accent)",color:"#fff",display:"flex",alignItems:"center",justifyContent:"center",gap:8,transition:"all 150ms ease"}}>
                <Play size={14} weight="fill"/>{running?"Running...":"Run Scenario"}
              </button>
            </div>
          </div>
        </div>

        {/* Right: results */}
        <div style={{display:"flex",flexDirection:"column",gap:20}}>
          {error&&<div style={{background:"rgba(239,68,68,.08)",border:"1px solid rgba(239,68,68,.2)",borderRadius:6,padding:"12px 16px",color:"#EF4444",fontSize:13}}>{error}</div>}
          {running&&(
            <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"40px",textAlign:"center"}}>
              <div style={{width:32,height:32,border:"2px solid var(--color-border)",borderTopColor:"var(--color-accent)",borderRadius:"50%",animation:"spin 1s linear infinite",margin:"0 auto 12px"}}/>
              <p style={{fontSize:14,color:"var(--color-text-secondary)"}}>Running {nPaths.toLocaleString()} Monte Carlo paths over {horizon} months...</p>
            </div>
          )}
          {result&&!running&&(
            <>
              <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
                <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:16}}><ChartLine size={16} color={getColor(result.scenario)}/><p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)"}}>Monte Carlo Fan Chart — Delinquency Rate</p></div>
                {result.summary?.fan_charts?.delinquency_rate ? (
                  <MonteCarloChart
                    paths={(result.summary.fan_charts.delinquency_rate.months ?? []).map((t,i)=>({
                      t,
                      p5: result.summary?.fan_charts?.delinquency_rate?.p5?.[i] ?? 0,
                      p50: result.summary?.fan_charts?.delinquency_rate?.p50?.[i] ?? 0,
                      p95: result.summary?.fan_charts?.delinquency_rate?.p95?.[i] ?? 0,
                    }))}
                    scenarioName={result.scenario}
                    color={getColor(result.scenario)}
                  />
                ) : (
                  <MonteCarloChart paths={result.paths ?? []} scenarioName={result.scenario} color={getColor(result.scenario)}/>
                )}
              </div>
              {result.summary&&(
                <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:16}}>
                  {[
                    {label:"Expected Loss",value:result.summary?.terminal?.expected_loss?.mean!=null?`${(result.summary.terminal.expected_loss.mean/1e6).toFixed(2)}M`:"—",color:"#EF4444"},
                    {label:"Terminal Default Rate",value:result.summary?.terminal?.default_rate?.mean!=null?`${(result.summary.terminal.default_rate.mean*100).toFixed(2)}%`:"—",color:"#F59E0B"},
                    {label:"Terminal Prepay Rate",value:result.summary?.terminal?.prepayment_rate?.mean!=null?`${(result.summary.terminal.prepayment_rate.mean*100).toFixed(1)}%`:"—",color:"#06B6D4"},
                  ].map(({label,value,color})=>(
                    <div key={label} style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"16px 20px"}}>
                      <p style={{fontSize:11,color:"var(--color-text-tertiary)",textTransform:"uppercase",letterSpacing:"0.06em",marginBottom:6}}>{label}</p>
                      <p style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:24,fontWeight:600,color}}>{value}</p>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
          {!result&&!running&&!error&&(
            <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"60px 40px",textAlign:"center"}}>
              <Flask size={36} color="var(--color-text-tertiary)" style={{margin:"0 auto 16px"}}/>
              <p style={{fontSize:15,fontWeight:500,color:"var(--color-text-primary)",marginBottom:6}}>Select a scenario and run</p>
              <p style={{fontSize:13,color:"var(--color-text-tertiary)"}}>Monte Carlo fan chart and stress metrics will appear here</p>
            </div>
          )}
        </div>
      </div>
      <style>{`@keyframes spin{to{transform:rotate(360deg);}}`}</style>
    </div>
  );
}
