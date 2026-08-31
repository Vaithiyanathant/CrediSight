"use client";
import { useEffect, useState } from "react";
import { submissionApi } from "@/lib/api";
import { SkeletonCard } from "@/components/ui/Skeleton";
import { CheckCircle, XCircle, WarningCircle, FileText, ArrowClockwise, DownloadSimple } from "@phosphor-icons/react";

// Errors/warnings from the API are either plain strings OR structured dicts like
// { check: "no_nan", column: "loan_id", n: 5 }
type ValidationItem = string | Record<string, unknown>;

/** Safely convert any error/warning item to a human-readable string */
function itemToString(item: ValidationItem): string {
  if (typeof item === "string") return item;
  return Object.entries(item)
    .map(([k, v]) => {
      if (Array.isArray(v)) return `${k}: [${v.join(", ")}]`;
      if (typeof v === "object" && v !== null) return `${k}: ${JSON.stringify(v)}`;
      return `${k}: ${v}`;
    })
    .join(" · ");
}

interface ValidationResult {
  valid: boolean;
  path: string;
  n_rows: number;
  n_loans: number;
  n_errors?: number;
  n_warnings?: number;
  errors?: ValidationItem[];
  warnings?: ValidationItem[];
  summary?: Record<string,unknown>;
  template?: string;
  elapsed_ms?: number;
}

interface SubmissionResult {
  valid: boolean;
  path: string;
  n_rows: number;
  n_loans: number;
  reporting_month: string;
  validation?: Record<string,unknown>;
  manifest?: Record<string,unknown>;
  preview?: Record<string,unknown>[];
  elapsed_ms: number;
}

export default function SubmissionPage() {
  const [validation, setValidation] = useState<ValidationResult|null>(null);
  const [generated,  setGenerated]  = useState<SubmissionResult|null>(null);
  const [validating, setValidating] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [valError,   setValError]   = useState<string|null>(null);
  const [genError,   setGenError]   = useState<string|null>(null);

  const validate = async () => {
    setValidating(true); setValError(null);
    try { const r = await submissionApi.validate(); setValidation(r.data); }
    catch (e:unknown) { setValError(e instanceof Error ? e.message : "Validation failed"); }
    finally { setValidating(false); }
  };

  const generate = async () => {
    setGenerating(true); setGenError(null);
    try { const r = await submissionApi.generate({}); setGenerated(r.data); validate(); }
    catch (e:unknown) { setGenError(e instanceof Error ? e.message : "Generation failed"); }
    finally { setGenerating(false); }
  };

  useEffect(() => { validate(); }, []);

  const StatusIcon = ({ ok, warn }:{ ok:boolean; warn?:boolean }) => ok
    ? <CheckCircle size={16} color="#22C55E" weight="fill"/>
    : warn ? <WarningCircle size={16} color="#F59E0B" weight="fill"/>
    : <XCircle size={16} color="#EF4444" weight="fill"/>;

  return (
    <div className="animate-fade-up" style={{maxWidth:1100,margin:"0 auto",display:"flex",flexDirection:"column",gap:24}}>

      {/* Header actions */}
      <div style={{display:"flex",alignItems:"center",gap:12}}>
        <FileText size={18} color="var(--color-accent)"/>
        <p style={{fontSize:16,fontWeight:600,color:"var(--color-text-primary)",flex:1}}>Submission</p>
        <button onClick={validate} disabled={validating} style={{display:"flex",alignItems:"center",gap:6,padding:"8px 16px",borderRadius:6,background:"transparent",border:"1px solid var(--color-border)",color:"var(--color-text-secondary)",fontSize:13,cursor:validating?"not-allowed":"pointer",opacity:validating?0.7:1,transition:"all 150ms ease"}}>
          <ArrowClockwise size={14}/>{validating?"Validating…":"Validate"}
        </button>
        <button onClick={generate} disabled={generating} style={{display:"flex",alignItems:"center",gap:6,padding:"8px 16px",borderRadius:6,background:"var(--color-accent)",border:"none",color:"#fff",fontSize:13,fontWeight:500,cursor:generating?"not-allowed":"pointer",opacity:generating?0.7:1,transition:"all 150ms ease"}}>
          <DownloadSimple size={14}/>{generating?"Generating…":"Regenerate"}
        </button>
      </div>

      {/* Validation card */}
      {validating&&<SkeletonCard rows={5}/>}
      {valError&&<div style={{background:"rgba(239,68,68,.08)",border:"1px solid rgba(239,68,68,.2)",borderRadius:8,padding:"14px 20px",color:"#EF4444",fontSize:13}}>{valError}</div>}
      {genError&&<div style={{background:"rgba(239,68,68,.08)",border:"1px solid rgba(239,68,68,.2)",borderRadius:8,padding:"14px 20px",color:"#EF4444",fontSize:13}}>{genError}</div>}

      {/* Status is carried by the heading icon and the summary row, not by a
          coloured left rail — the reference design uses one flat hairline
          border on every card regardless of state. */}
      {validation&&!validating&&(
        <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
          <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:16}}>
            <StatusIcon ok={validation.valid}/>
            <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)"}}>{validation.valid?"Submission Valid":"Submission Has Issues"}</p>
            {validation.elapsed_ms!==undefined&&<span style={{marginLeft:"auto",fontSize:11,color:"var(--color-text-tertiary)",fontFamily:"var(--font-geist-mono),monospace"}}>{validation.elapsed_ms.toFixed(0)}ms</span>}
          </div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:16,marginBottom:20}}>
            {[
              {label:"Rows",value:validation.n_rows.toLocaleString(),color:"var(--color-text-primary)"},
              {label:"Loans",value:validation.n_loans.toLocaleString(),color:"var(--color-text-primary)"},
              {label:"Errors",value:(validation.n_errors??0).toString(),color:(validation.n_errors??0)>0?"#EF4444":"#22C55E"},
              {label:"Warnings",value:(validation.n_warnings??0).toString(),color:(validation.n_warnings??0)>0?"#F59E0B":"#22C55E"},
            ].map(({label,value,color})=>(
              <div key={label} style={{background:"var(--color-bg-elevated)",borderRadius:6,padding:"12px 16px"}}>
                <p style={{fontSize:11,color:"var(--color-text-tertiary)",textTransform:"uppercase",letterSpacing:"0.06em",marginBottom:4}}>{label}</p>
                <p style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:22,fontWeight:600,color}}>{value}</p>
              </div>
            ))}
          </div>
          <p style={{fontSize:11,color:"var(--color-text-tertiary)",marginBottom:4}}>Path</p>
          <p style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:"var(--color-text-secondary)",background:"var(--color-bg-elevated)",borderRadius:4,padding:"6px 10px",marginBottom:16}}>{validation.path}</p>

          {/* Errors */}
          {validation.errors&&validation.errors.length>0&&(
            <div style={{marginBottom:16}}>
              <p style={{fontSize:12,fontWeight:500,color:"#EF4444",marginBottom:8,display:"flex",alignItems:"center",gap:6}}><XCircle size={13} color="#EF4444"/>Errors ({validation.errors.length})</p>
              <div style={{display:"flex",flexDirection:"column",gap:4}}>
                {validation.errors.map((item,i)=>(
                  <div key={i} style={{fontSize:12,color:"var(--color-text-secondary)",background:"rgba(239,68,68,.06)",border:"1px solid rgba(239,68,68,.15)",borderRadius:4,padding:"6px 10px",fontFamily:"var(--font-geist-mono),monospace"}}>{itemToString(item)}</div>
                ))}
              </div>
            </div>
          )}

          {/* Warnings */}
          {validation.warnings&&validation.warnings.length>0&&(
            <div style={{marginBottom:16}}>
              <p style={{fontSize:12,fontWeight:500,color:"#F59E0B",marginBottom:8,display:"flex",alignItems:"center",gap:6}}><WarningCircle size={13} color="#F59E0B"/>Warnings ({validation.warnings.length})</p>
              <div style={{display:"flex",flexDirection:"column",gap:4}}>
                {validation.warnings.map((item,i)=>(
                  <div key={i} style={{fontSize:12,color:"var(--color-text-secondary)",background:"rgba(245,158,11,.06)",border:"1px solid rgba(245,158,11,.15)",borderRadius:4,padding:"6px 10px",fontFamily:"var(--font-geist-mono),monospace"}}>{itemToString(item)}</div>
                ))}
              </div>
            </div>
          )}

          {validation.valid&&!validation.errors?.length&&!validation.warnings?.length&&(
            <div style={{display:"flex",alignItems:"center",gap:8,background:"rgba(34,197,94,.06)",border:"1px solid rgba(34,197,94,.2)",borderRadius:6,padding:"10px 14px"}}>
              <CheckCircle size={14} color="#22C55E" weight="fill"/>
              <span style={{fontSize:13,color:"#22C55E"}}>All checks passed. Submission is ready.</span>
            </div>
          )}
        </div>
      )}

      {/* Generated preview */}
      {generated&&generated.preview&&generated.preview.length>0&&(
        <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8}}>
          <div style={{padding:"14px 20px",borderBottom:"1px solid var(--color-border)",display:"flex",alignItems:"center",gap:8}}>
            <FileText size={14} color="var(--color-accent)"/>
            <p style={{fontSize:13,fontWeight:500,color:"var(--color-text-primary)",flex:1}}>Preview (first 5 rows)</p>
            <span style={{fontSize:11,color:"var(--color-text-tertiary)",fontFamily:"var(--font-geist-mono),monospace"}}>{generated.n_rows.toLocaleString()} total rows · {generated.n_loans.toLocaleString()} loans · {generated.reporting_month}</span>
          </div>
          <div style={{overflowX:"auto"}}>
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
              <thead><tr>{Object.keys(generated.preview[0]).map(k=>(
                <th key={k} style={{fontSize:10,letterSpacing:"0.06em",textTransform:"uppercase",color:"var(--color-text-tertiary)",fontWeight:400,textAlign:"left",padding:"8px 12px",borderBottom:"1px solid var(--color-border)",whiteSpace:"nowrap"}}>{k}</th>
              ))}</tr></thead>
              <tbody>{generated.preview.map((row,i)=>(
                <tr key={i} style={{borderBottom:"1px solid var(--color-border)"}}>
                  {Object.values(row).map((v,j)=>(
                    <td key={j} style={{padding:"8px 12px",fontFamily:"var(--font-geist-mono),monospace",color:"var(--color-text-secondary)",whiteSpace:"nowrap"}}>{v===null||v===undefined?"—":typeof v==="object"?JSON.stringify(v):String(v)}</td>
                  ))}
                </tr>
              ))}</tbody>
            </table>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!validation&&!validating&&(
        <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"60px 40px",textAlign:"center"}}>
          <FileText size={36} color="var(--color-text-tertiary)" style={{margin:"0 auto 16px"}}/>
          <p style={{fontSize:15,fontWeight:500,color:"var(--color-text-primary)",marginBottom:6}}>Submission Manager</p>
          <p style={{fontSize:13,color:"var(--color-text-tertiary)"}}>Validate or regenerate your submission.csv file</p>
        </div>
      )}
    </div>
  );
}
