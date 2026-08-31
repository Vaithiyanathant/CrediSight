"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { portfolioApi } from "@/lib/api";
import { MagnifyingGlass, ArrowRight, Lightning, WarningOctagon } from "@phosphor-icons/react";
import type { WatchlistEntry } from "@/types";

const LOAN_ID_RE = /^LN\d{7}$/;

interface Sample { loan_id: string; status: string; prob: number }

export default function LoanSearchPage(){
  const [id,setId]=useState("");
  const [samples,setSamples]=useState<Sample[]>([]);
  const [loadingSamples,setLoadingSamples]=useState(true);
  const router=useRouter();

  // Real loans to try, not invented IDs: the top of the watchlist is the most
  // interesting thing to open on this screen (highest expected loss), and
  // every ID is guaranteed to exist so a sample click can never 404.
  useEffect(()=>{
    let cancelled=false;
    portfolioApi.watchlist({n:8})
      .then(r=>{ if(!cancelled) setSamples(((r.data.entries??[]) as WatchlistEntry[]).map(e=>({
        loan_id:e.loan_id, status:e.current_status, prob:e.prob_next_12m_default??0,
      }))); })
      .catch(()=>{ if(!cancelled) setSamples([]); })
      .finally(()=>{ if(!cancelled) setLoadingSamples(false); });
    return ()=>{cancelled=true;};
  },[]);

  const trimmed=id.trim().toUpperCase();
  const valid=LOAN_ID_RE.test(trimmed);
  // Only surface the format hint once they've actually typed something wrong —
  // an empty field is not an error state.
  const showHint=trimmed.length>0 && !valid;
  const go=(explicit?:string)=>{
    const target=(explicit??trimmed).toUpperCase();
    if(LOAN_ID_RE.test(target)) router.push(`/loan/${target}`);
  };

  return(
    <div className="animate-fade-up" style={{maxWidth:620,margin:"64px auto"}}>
      <div style={{textAlign:"center"}}>
        <div style={{width:52,height:52,borderRadius:12,background:"var(--color-accent)",display:"flex",alignItems:"center",justifyContent:"center",margin:"0 auto 18px"}}>
          <MagnifyingGlass size={22} color="var(--color-text-inverse)"/>
        </div>
        <h2 style={{fontSize:22,fontWeight:600,color:"var(--color-text-primary)",marginBottom:8}}>Loan 360</h2>
        <p style={{fontSize:14,color:"var(--color-text-tertiary)",marginBottom:24,lineHeight:1.6}}>
          Enter a loan ID to view its full risk profile, SHAP drivers, survival curve and reviewer decision.
        </p>
      </div>

      <div style={{display:"flex",gap:10}}>
        <div style={{flex:1,minWidth:0}}>
          <input
            value={id}
            onChange={e=>setId(e.target.value)}
            onKeyDown={e=>{if(e.key==="Enter")go();}}
            placeholder="LN0000001"
            aria-label="Loan ID"
            aria-invalid={showHint}
            style={{width:"100%",background:"var(--color-bg-surface)",border:`1px solid ${showHint?"var(--color-risk-high)":"var(--color-border)"}`,borderRadius:8,padding:"10px 14px",fontSize:14,color:"var(--color-text-primary)",fontFamily:"var(--font-geist-mono),monospace",letterSpacing:"0.04em"}}
          />
        </div>
        <button onClick={()=>go()} disabled={!valid} style={{padding:"10px 20px",borderRadius:8,background:valid?"var(--color-accent)":"var(--color-bg-subtle)",border:"none",color:valid?"var(--color-text-inverse)":"var(--color-text-tertiary)",fontSize:13,fontWeight:500,cursor:valid?"pointer":"not-allowed",display:"flex",alignItems:"center",gap:8,flexShrink:0,transition:"background 150ms ease"}}>
          View<ArrowRight size={14}/>
        </button>
      </div>
      <p style={{fontSize:12,minHeight:18,marginTop:8,color:showHint?"var(--color-risk-high)":"var(--color-text-tertiary)",fontFamily:"var(--font-geist-mono),monospace"}}>
        {showHint?"Format: LN + 7 digits, e.g. LN0000001":" "}
      </p>

      {/* Sample loans — click to fill the field and open */}
      <div style={{marginTop:20,background:"var(--color-bg-surface)",borderRadius:10,padding:"16px 18px",boxShadow:"var(--shadow-sm)"}}>
        <div style={{display:"flex",alignItems:"center",gap:7,marginBottom:12}}>
          <Lightning size={14} color="var(--color-accent)" weight="fill"/>
          <p style={{fontSize:13,fontWeight:600,color:"var(--color-text-primary)",margin:0}}>Try a loan</p>
          <span style={{fontSize:11,color:"var(--color-text-tertiary)",marginLeft:"auto"}}>highest expected loss</span>
        </div>

        {loadingSamples?(
          <div style={{display:"flex",flexWrap:"wrap",gap:8}}>
            {Array.from({length:8}).map((_,i)=>(<div key={i} className="skeleton" style={{height:30,width:132,borderRadius:999}}/>))}
          </div>
        ):samples.length===0?(
          <p style={{fontSize:12,color:"var(--color-text-tertiary)",margin:0}}>
            Sample loans unavailable — type any ID in the form LN0000001.
          </p>
        ):(
          <div style={{display:"flex",flexWrap:"wrap",gap:8}}>
            {samples.map(s=>{
              const hot=s.prob>=0.5;
              return (
                <button
                  key={s.loan_id}
                  onClick={()=>{ setId(s.loan_id); go(s.loan_id); }}
                  title={`${s.status} · ${(s.prob*100).toFixed(1)}% 12m default`}
                  style={{display:"inline-flex",alignItems:"center",gap:7,padding:"6px 11px",borderRadius:999,cursor:"pointer",
                    background:"var(--color-bg-subtle)",border:`1px solid ${hot?"hsl(354 80% 65% / .35)":"var(--color-border)"}`,
                    color:"var(--color-text-secondary)",fontSize:12,fontFamily:"var(--font-geist-mono),monospace",
                    transition:"border-color 150ms ease, color 150ms ease, background 150ms ease"}}
                  onMouseEnter={e=>{const el=e.currentTarget;el.style.borderColor="var(--color-accent)";el.style.color="var(--color-text-primary)";}}
                  onMouseLeave={e=>{const el=e.currentTarget;el.style.borderColor=hot?"hsl(354 80% 65% / .35)":"var(--color-border)";el.style.color="var(--color-text-secondary)";}}
                >
                  {hot&&<WarningOctagon size={12} color="var(--color-risk-high)" weight="fill"/>}
                  {s.loan_id}
                  <span style={{color:hot?"var(--color-risk-high)":"var(--color-text-tertiary)",fontSize:11}}>
                    {(s.prob*100).toFixed(0)}%
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
