"use client";
import { useState, useRef, useEffect } from "react";
import { copilotApi } from "@/lib/api";
import { AiBanner } from "@/components/ui/AiBanner";
import { PromptLogEntryCard } from "@/components/ui/PromptLogEntryCard";
import { RiskBadge } from "@/components/ui/RiskBadge";
import type { CopilotResponseModel, PromptLogEntry } from "@/types";
import { Brain, PaperPlaneTilt, Robot, Clock } from "@phosphor-icons/react";

interface Message { role:"user"|"assistant"; content:string; response?:CopilotResponseModel; }

const QUICK=["What is the current portfolio default rate?","Which loans have the highest anomaly scores?","Summarize the top 5 risk drivers across the portfolio.","What is the expected loss under the adverse scenario?","Which states have the highest delinquency rates?"];

export default function CopilotPage(){
  const [messages,setMessages]=useState<Message[]>([]);
  const [input,setInput]=useState("");
  const [loanId,setLoanId]=useState("");
  const [loading,setLoading]=useState(false);
  const [log,setLog]=useState<PromptLogEntry[]>([]);
  const [logLoading,setLogLoading]=useState(true);
  const bottomRef=useRef<HTMLDivElement>(null);

  useEffect(()=>{ bottomRef.current?.scrollIntoView({behavior:"smooth"}); },[messages]);

  useEffect(()=>{
    copilotApi.log({limit:20}).then(r=>setLog(r.data.entries??[])).catch(()=>{}).finally(()=>setLogLoading(false));
  },[]);

  const send=async(q?:string)=>{
    const question=q||input.trim(); if(!question) return;
    setInput(""); setLoading(true);
    const userMsg:Message={role:"user",content:question};
    setMessages(prev=>[...prev,userMsg]);
    try{
      const r=await copilotApi.ask({question,loan_id:loanId||undefined});
      setMessages(prev=>[...prev,{role:"assistant",content:r.data.answer,response:r.data}]);
    }catch(e:unknown){
      setMessages(prev=>[...prev,{role:"assistant",content:`Error: ${e instanceof Error?e.message:"Request failed"}`,response:undefined}]);
    }finally{ setLoading(false); }
  };

  return(
    <div className="animate-fade-up" style={{maxWidth:1200,margin:"0 auto",display:"grid",gridTemplateColumns:"1fr 300px",gap:20,alignItems:"start"}}>
      {/* Chat */}
      <div style={{display:"flex",flexDirection:"column",gap:0,background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,overflow:"hidden"}}>
        {/* Header */}
        <div style={{padding:"14px 20px",borderBottom:"1px solid var(--color-border)",display:"flex",alignItems:"center",gap:10}}>
          <Brain size={16} color="var(--color-accent)"/>
          <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)",flex:1}}>AI Copilot</p>
          <div style={{padding:"3px 8px",background:"var(--color-ai-banner)",borderRadius:4,fontSize:10,color:"var(--color-ai-banner-text)",letterSpacing:"0.06em",fontWeight:500}}>GOVERNED</div>
        </div>

        {/* Messages */}
        <div style={{minHeight:420,maxHeight:520,overflowY:"auto",padding:"16px 20px",display:"flex",flexDirection:"column",gap:16}}>
          {messages.length===0&&(
            <div style={{textAlign:"center",paddingBlock:40}}>
              <Robot size={36} color="var(--color-text-tertiary)" style={{margin:"0 auto 12px"}}/>
              <p style={{fontSize:15,fontWeight:500,color:"var(--color-text-primary)",marginBottom:6}}>Grounded portfolio intelligence</p>
              <p style={{fontSize:13,color:"var(--color-text-tertiary)",marginBottom:24}}>All answers are RAG-grounded. Numbers are verified by the numeric verifier.</p>
              <div style={{display:"flex",flexWrap:"wrap",gap:8,justifyContent:"center"}}>
                {QUICK.map(q=>(
                  <button key={q} onClick={()=>send(q)} style={{fontSize:12,padding:"6px 12px",borderRadius:6,border:"1px solid var(--color-border)",background:"var(--color-bg-elevated)",color:"var(--color-text-secondary)",cursor:"pointer",textAlign:"left",transition:"all 150ms ease"}} onMouseEnter={e=>{(e.currentTarget as HTMLElement).style.borderColor="var(--color-accent)";(e.currentTarget as HTMLElement).style.color="var(--color-text-primary)";}} onMouseLeave={e=>{(e.currentTarget as HTMLElement).style.borderColor="var(--color-border)";(e.currentTarget as HTMLElement).style.color="var(--color-text-secondary)";}}>{q}</button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m,i)=>(
            <div key={i} style={{display:"flex",flexDirection:"column",gap:8,alignItems:m.role==="user"?"flex-end":"flex-start"}}>
              <div style={{maxWidth:"85%",background:m.role==="user"?"var(--color-accent)":"var(--color-bg-elevated)",borderRadius:m.role==="user"?"12px 12px 2px 12px":"12px 12px 12px 2px",padding:"10px 14px"}}>
                <p style={{fontSize:13,color:m.role==="user"?"#fff":"var(--color-text-primary)",lineHeight:1.6,whiteSpace:"pre-wrap"}}>{m.content}</p>
              </div>
              {m.response&&(
                <div style={{maxWidth:"85%",display:"flex",flexDirection:"column",gap:8}}>
                  <AiBanner verdict={m.response.verifier?.verdict} model={m.response.model} latencyMs={m.response.latency_ms}/>
                  {m.response.citations&&m.response.citations.length>0&&(
                    <div style={{display:"flex",flexWrap:"wrap",gap:8}}>
                      {m.response.citations.map((c,ci)=>{
                        const text = typeof c === "string" ? c : ((c as {chunk?:string;text?:string;source?:string}).chunk ?? (c as {text?:string}).text ?? JSON.stringify(c));
                        const display = text.slice(0,60).replace(/\n/g," ");
                        return (
                          <span key={ci} title={text.slice(0,200)} style={{fontSize:10,padding:"2px 7px",background:"rgba(99,102,241,.1)",border:"1px solid rgba(99,102,241,.2)",borderRadius:4,color:"var(--color-accent)",fontFamily:"var(--font-geist-mono),monospace",cursor:"help",maxWidth:200,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>
                            [{ci+1}] {display}
                          </span>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {loading&&(
            <div style={{display:"flex",alignItems:"center",gap:8}}>
              <div style={{display:"flex",gap:4}}>
                {[0,1,2].map(i=>(<div key={i} style={{width:6,height:6,borderRadius:"50%",background:"var(--color-accent)",animation:`bounce 1.2s ease-in-out ${i*0.2}s infinite`}}/>))}
              </div>
              <span style={{fontSize:12,color:"var(--color-text-tertiary)"}}>Verifying answer...</span>
            </div>
          )}
          <div ref={bottomRef}/>
        </div>

        {/* Input */}
        <div style={{padding:"14px 20px",borderTop:"1px solid var(--color-border)",display:"flex",flexDirection:"column",gap:8}}>
          <div style={{display:"flex",gap:8,alignItems:"center"}}>
            <input value={loanId} onChange={e=>setLoanId(e.target.value)} placeholder="Loan ID (optional)" style={{width:160,background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)",borderRadius:6,padding:"7px 10px",fontSize:12,color:"var(--color-text-primary)",fontFamily:"var(--font-geist-mono),monospace"}}/>
            <input value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}}} placeholder="Ask about portfolio risk, anomalies, compliance..." style={{flex:1,background:"var(--color-bg-subtle)",border:"1px solid var(--color-border)",borderRadius:6,padding:"7px 12px",fontSize:13,color:"var(--color-text-primary)"}} disabled={loading}/>
            <button onClick={()=>send()} disabled={!input.trim()||loading} style={{width:36,height:36,borderRadius:6,background:input.trim()&&!loading?"var(--color-accent)":"var(--color-bg-subtle)",border:"none",cursor:input.trim()&&!loading?"pointer":"not-allowed",display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0,transition:"all 150ms ease"}}>
              <PaperPlaneTilt size={15} color={input.trim()&&!loading?"#fff":"var(--color-text-tertiary)"}/>
            </button>
          </div>
        </div>
      </div>

      {/* Prompt Log */}
      <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,position:"sticky",top:"calc(var(--header-height) + 32px)"}}>
        <div style={{padding:"14px 16px",borderBottom:"1px solid var(--color-border)",display:"flex",alignItems:"center",gap:8}}>
          <Clock size={14} color="var(--color-text-tertiary)"/>
          <p style={{fontSize:13,fontWeight:500,color:"var(--color-text-primary)"}}>Prompt Log</p>
        </div>
        <div style={{maxHeight:560,overflowY:"auto"}}>
          {logLoading?<div style={{padding:"16px"}}>{[0,1,2,3].map(i=>(<div key={i} className="skeleton" style={{height:50,marginBottom:8,borderRadius:6}}/>))}</div>:
          log.length===0?<p style={{padding:"20px 16px",fontSize:13,color:"var(--color-text-tertiary)",textAlign:"center"}}>No logged prompts yet.</p>:
          log.map(entry=>(<PromptLogEntryCard key={entry.id} entry={entry}/>))}
        </div>
      </div>
      <style>{`@keyframes bounce{0%,80%,100%{transform:scale(0);}40%{transform:scale(1.0);}}`}</style>
    </div>
  );
}
