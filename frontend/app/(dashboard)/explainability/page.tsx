"use client";
import { useEffect, useState } from "react";
import { explainApi } from "@/lib/api";
import { SkeletonCard } from "@/components/ui/Skeleton";
import type { GlobalExplainResponse } from "@/types";
import { Pulse, ChartBar } from "@phosphor-icons/react";
import ReactECharts from "echarts-for-react";

const HEADS=["next_12m_default","next_3m_delinquency","next_6m_delinquency","next_12m_prepayment","next_state","exception_required"];
const FC:Record<string,string>={static:"#6366F1",balance:"#06B6D4",delinquency:"#EF4444",prepay:"#22C55E",cohort:"#F59E0B",other:"#A1A1AA"};

type FamArr = Array<{family:string;share?:number;total_mean_abs_shap?:number}>;

export default function ExplainabilityPage(){
  const [head,setHead]=useState("next_12m_default");
  const [data,setData]=useState<GlobalExplainResponse|null>(null);
  const [loading,setLoading]=useState(false);

  useEffect(()=>{
    setLoading(true); setData(null);
    explainApi.global({head,top_k:20}).then(r=>setData(r.data)).catch(()=>{}).finally(()=>setLoading(false));
  },[head]);

  const importance = data ? (data.global_importance ?? data.importance ?? []) : [];
  const famArr = data?.family_attribution;
  const familyAttr: Record<string,number> = famArr
    ? Array.isArray(famArr)
      ? Object.fromEntries((famArr as FamArr).map(f=>[f.family, f.share??0]))
      : (famArr as Record<string,number>)
    : {};

  const barOption = importance.length>0?{
    backgroundColor:"transparent",
    tooltip:{trigger:"axis",axisPointer:{type:"shadow"},backgroundColor:"#1A1A1F",borderColor:"#2E2E36",textStyle:{color:"#FAFAFA",fontSize:12}},
    grid:{left:12,right:60,top:8,bottom:8,containLabel:true},
    xAxis:{type:"value",axisLabel:{color:"#71717A",fontSize:10,fontFamily:"var(--font-geist-mono)"},splitLine:{lineStyle:{color:"#2E2E3630"}}},
    yAxis:{type:"category",data:importance.slice(0,15).map(f=>f.feature).reverse(),axisLabel:{color:"#A1A1AA",fontSize:11,fontFamily:"var(--font-geist-sans)",width:150,overflow:"truncate"},axisLine:{lineStyle:{color:"#2E2E36"}},axisTick:{show:false}},
    series:[{type:"bar",barMaxWidth:24,borderRadius:3,data:importance.slice(0,15).map(f=>({value:f.mean_abs_shap??f.shap_mean_abs??0,itemStyle:{color:FC[f.family??"other"]??"#6366F1"}})).reverse(),label:{show:true,position:"right",color:"#A1A1AA",fontFamily:"var(--font-geist-mono)",fontSize:10,formatter:(p:{value:number})=>p.value.toFixed(4)}}],
  }:null;

  const pieData=Object.entries(familyAttr).filter(([,v])=>v>0);
  const pieOption=pieData.length>0?{
    backgroundColor:"transparent",
    tooltip:{trigger:"item",backgroundColor:"#1A1A1F",borderColor:"#2E2E36",textStyle:{color:"#FAFAFA",fontSize:12}},
    legend:{orient:"vertical",right:10,top:"center",textStyle:{color:"#A1A1AA",fontSize:11}},
    series:[{type:"pie",radius:["45%","70%"],center:["38%","50%"],avoidLabelOverlap:false,label:{show:false},data:pieData.map(([k,v])=>({name:k,value:v,itemStyle:{color:FC[k]??"#6366F1"}})),emphasis:{itemStyle:{shadowBlur:8,shadowColor:"rgba(0,0,0,.5)"}}}]
  }:null;

  return(
    <div className="animate-fade-up" style={{maxWidth:1200,margin:"0 auto",display:"flex",flexDirection:"column",gap:20}}>
      <div style={{display:"flex",alignItems:"center",gap:12}}>
        <Pulse size={16} color="var(--color-accent)"/>
        <p style={{fontSize:14,fontWeight:500,color:"var(--color-text-primary)"}}>Global Feature Importance</p>
        <div style={{display:"flex",gap:6,marginLeft:"auto",flexWrap:"wrap"}}>
          {HEADS.map(h=>(
            <button key={h} onClick={()=>setHead(h)} style={{padding:"5px 10px",borderRadius:5,fontSize:11,cursor:"pointer",border:`1px solid ${head===h?"var(--color-accent)":"var(--color-border)"}`,background:head===h?"rgba(99,102,241,.1)":"transparent",color:head===h?"var(--color-accent)":"var(--color-text-secondary)",fontFamily:"var(--font-geist-mono),monospace",transition:"all 150ms ease"}}>{h.replace(/_/g," ")}</button>
          ))}
        </div>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"2fr 1fr",gap:20}}>
        <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:12}}><ChartBar size={14} color="var(--color-accent)"/><p style={{fontSize:13,fontWeight:500,color:"var(--color-text-primary)"}}>Top 15 Features by Mean |SHAP|</p></div>
          {loading?<SkeletonCard rows={6}/>:barOption?<ReactECharts option={barOption} style={{height:460}}/>:<p style={{fontSize:13,color:"var(--color-text-tertiary)",padding:20,textAlign:"center"}}>No SHAP data available for this head.</p>}
        </div>
        <div style={{display:"flex",flexDirection:"column",gap:20}}>
          <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
            <p style={{fontSize:13,fontWeight:500,color:"var(--color-text-primary)",marginBottom:12}}>Family Attribution</p>
            {loading?<SkeletonCard rows={3}/>:pieOption?<ReactECharts option={pieOption} style={{height:200}}/>:<p style={{fontSize:13,color:"var(--color-text-tertiary)"}}>No data.</p>}
          </div>
          <div style={{background:"var(--color-bg-surface)",border:"1px solid var(--color-border)",borderRadius:8,padding:"20px 24px"}}>
            <p style={{fontSize:13,fontWeight:500,color:"var(--color-text-primary)",marginBottom:12}}>Feature Families</p>
            <div style={{display:"flex",flexDirection:"column",gap:6}}>
              {Object.entries(FC).map(([fam,color])=>(
                <div key={fam} style={{display:"flex",alignItems:"center",gap:8}}>
                  <div style={{width:10,height:10,borderRadius:2,background:color,flexShrink:0}}/>
                  <span style={{fontSize:12,color:"var(--color-text-secondary)",textTransform:"capitalize"}}>{fam}</span>
                  {familyAttr[fam]!=null&&<span style={{fontFamily:"var(--font-geist-mono),monospace",fontSize:12,color:"var(--color-text-primary)",marginLeft:"auto"}}>{((familyAttr[fam]??0)*100).toFixed(1)}%</span>}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
