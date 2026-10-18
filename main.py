import os
import re
from typing import List
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="TruthLens AI",
    description="Evidence-grounded LLM hallucination screening API",
    version="1.0.0",
)

def split_claims(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text) if p.strip()]

def words(text: str) -> set:
    return set(re.findall(r"\b[\w'-]+\b", text.lower()))

def score_claim(claim: str, evidence: List[str], threshold: float = 0.35):
    if not evidence:
        return {
            "claim": claim, "verdict": "INSUFFICIENT_EVIDENCE", "confidence": 0.0,
            "evidence": None,
            "explanation": "No reference evidence was provided. This does not mean the claim is false.",
            "method": "no_evidence"
        }

    claim_words = words(claim)
    ranked = []
    for passage in evidence:
        passage_words = words(passage)
        overlap = len(claim_words & passage_words) / max(1, len(claim_words))
        ranked.append((overlap, passage))
    similarity, best = max(ranked, key=lambda x: x[0])

    # Transparent baseline: exact negation cues + high lexical overlap can flag a possible contradiction.
    negation = (" not ", " never ", " no ", " isn't ", " doesn't ", " cannot ", " wasn't ", " aren't ")
    claim_padded, evidence_padded = f" {claim.lower()} ", f" {best.lower()} "
    claim_neg = any(x in claim_padded for x in negation)
    evidence_neg = any(x in evidence_padded for x in negation)

    if similarity >= threshold and claim_neg != evidence_neg:
        verdict = "POSSIBLE_CONTRADICTION"
        explanation = (
            "The claim and evidence have substantial word overlap but differ in a negation cue. "
            "This is only a heuristic flag; review the wording and context."
        )
    elif similarity >= threshold:
        verdict = "POSSIBLY_SUPPORTED"
        explanation = (
            "Matching terms were found in the evidence. Lexical overlap does not establish that "
            "the evidence entails the claim."
        )
    else:
        verdict = "INSUFFICIENT_EVIDENCE"
        explanation = (
            "No sufficiently overlapping evidence was found. This is not proof that the claim is false."
        )

    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": round(min(similarity, 1.0), 4),
        "similarity": round(similarity, 4),
        "evidence": best,
        "explanation": explanation,
        "method": "lexical_baseline"
    }

def analyze(answer: str, evidence: List[str], threshold: float = 0.35):
    claims = split_claims(answer)
    passages = [re.sub(r"\s+", " ", p).strip() for p in evidence if p.strip()]
    results = [score_claim(c, passages, threshold) for c in claims]
    total = len(results)
    counts = {v: sum(r["verdict"] == v for r in results) for v in [
        "POSSIBLY_SUPPORTED", "POSSIBLE_CONTRADICTION", "INSUFFICIENT_EVIDENCE"
    ]}
    return {
        "summary": {
            "total_claims": total,
            **{k.lower(): v for k, v in counts.items()},
            "evidence_coverage": round(sum(bool(r.get("evidence")) for r in results) / total, 4) if total else 0.0,
        },
        "results": results,
        "limitations": [
            "This Render-ready version uses a lexical baseline, not a trained NLI model.",
            "Similarity and word overlap are not calibrated probabilities of truth.",
            "Contradiction flags are heuristic and may be wrong; human review is required.",
            "Results are only as good as the evidence supplied by the user."
        ]
    }

class AnalyzeRequest(BaseModel):
    answer: str = Field(..., min_length=1, max_length=30000)
    evidence: List[str] = Field(default_factory=list, max_length=100)
    similarity_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

@app.get("/health")
def health():
    return {"status": "ok", "service": "truthlens-ai"}

@app.post("/analyze")
def analyze_endpoint(request: AnalyzeRequest):
    return analyze(request.answer, request.evidence, request.similarity_threshold)

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML
HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TruthLens AI — Hallucination Screening</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#131b30;--line:#293653;--text:#edf2ff;--muted:#a7b4ce;--accent:#8b9cff;--green:#77e0b5;--red:#ff9eaa;--yellow:#ffd580}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at top,#172447 0%,var(--bg) 55%);color:var(--text);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1100px;margin:auto;padding:34px 20px 60px}.brand{display:flex;align-items:center;gap:12px}.logo{width:42px;height:42px;border-radius:13px;background:#29375e;display:grid;place-items:center;font-size:23px}
h1{font-size:clamp(2rem,5vw,3.2rem);line-height:1.1;margin:28px 0 12px;letter-spacing:-.04em}.grad{color:#aab6ff}.lead{color:var(--muted);max-width:720px}
.panel{background:rgba(19,27,48,.92);border:1px solid var(--line);border-radius:18px;padding:22px;margin-top:24px;box-shadow:0 20px 50px #0002}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.field{display:flex;flex-direction:column;gap:8px}label{font-weight:650;font-size:.94rem}
textarea{width:100%;min-height:190px;resize:vertical;background:#0b1224;border:1px solid #344363;border-radius:12px;color:var(--text);padding:13px;font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace}textarea:focus{outline:2px solid var(--accent);border-color:transparent}
small,.muted{color:var(--muted)}.actions{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:18px}
button{border:0;border-radius:11px;padding:12px 18px;font-weight:750;cursor:pointer;background:#9eacff;color:#11182d}button:disabled{opacity:.6;cursor:wait}.secondary{background:#263452;color:var(--text)}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.metric{border:1px solid var(--line);background:#0e1629;border-radius:13px;padding:15px}.metric strong{display:block;font-size:1.7rem}.metric span{color:var(--muted);font-size:.85rem}
.result{border:1px solid var(--line);border-radius:13px;padding:16px;margin-top:12px;background:#0e1629}.topline{display:flex;justify-content:space-between;gap:10px;align-items:start}.tag{font-size:.75rem;font-weight:800;padding:4px 8px;border-radius:8px;background:#293653;white-space:nowrap}.supported{color:var(--green)}.contradiction{color:var(--red)}.insufficient{color:var(--yellow)}.evidence{border-left:3px solid #6477d8;padding:8px 12px;margin-top:12px;color:#cbd6f2;background:#131e37;white-space:pre-wrap}
.notice{font-size:.86rem;color:var(--muted);border-top:1px solid var(--line);padding-top:15px;margin-top:20px}.error{color:var(--red)}.hidden{display:none}
@media(max-width:720px){.grid{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.panel{padding:16px}}
</style>
</head>
<body><main class="wrap">
<div class="brand"><div class="logo">⌕</div><div><strong>TRUTHLENS AI</strong><div class="muted" style="font-size:.78rem">EVIDENCE-GROUNDED AI QUALITY</div></div></div>
<h1>Can you <span class="grad">trust the answer?</span></h1>
<p class="lead">Screen LLM responses claim by claim. Provide reference passages, inspect matching evidence, and flag claims that need review.</p>
<section class="panel">
<div class="grid">
<div class="field"><label for="answer">LLM-generated answer</label><textarea id="answer" placeholder="Paste an AI-generated answer here..."></textarea><small>Each sentence is treated as a candidate claim.</small></div>
<div class="field"><label for="evidence">Reference evidence</label><textarea id="evidence" placeholder="Paste trusted source passages. Put each passage on a separate line..."></textarea><small>Use reliable source text relevant to the answer.</small></div>
</div>
<div class="actions"><button id="run">Analyze answer →</button><button class="secondary" id="demo">Load demo</button><span id="status" class="muted" role="status"></span></div>
</section>
<section id="output" class="panel hidden" aria-live="polite">
<h2>Verification report</h2><div id="metrics" class="metrics"></div><div id="results"></div>
<div class="actions"><button class="secondary" id="download">Download JSON report</button></div>
<div class="notice" id="limitations"></div>
</section>
<p class="notice">TruthLens is a screening aid, not a truth oracle. This deployable starter uses lexical overlap and simple negation heuristics; it does not run a trained NLI model. Unsupported does not mean false. Review all high-impact results.</p>
</main>
<script>
let lastReport=null;
const $=id=>document.getElementById(id);
$("demo").onclick=()=>{$("answer").value="Water freezes at 0 degrees Celsius. The Sun does not revolve around Earth."; $("evidence").value="Pure water freezes at 0 degrees Celsius under standard atmospheric pressure.\\nEarth revolves around the Sun.";};
$("run").onclick=async()=>{
 const answer=$("answer").value.trim();
 const evidence=$("evidence").value.split("\\n").map(x=>x.trim()).filter(Boolean);
 if(!answer){$("status").textContent="Please enter an answer first.";return;}
 $("run").disabled=true;$("status").textContent="Analyzing claims…";
 try{
  const r=await fetch("/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({answer,evidence})});
  if(!r.ok)throw new Error((await r.text()).slice(0,250));
  const data=await r.json();lastReport=data;$("output").classList.remove("hidden");
  const s=data.summary;
  const cards=[["Claims",s.total_claims],["Possibly supported",s.possibly_supported],["Possible contradictions",s.possible_contradiction],["Evidence coverage",Math.round(s.evidence_coverage*100)+"%"]];
  $("metrics").innerHTML=cards.map(([label,val])=>`<div class="metric"><strong>${val}</strong><span>${label}</span></div>`).join("");
  $("results").innerHTML=data.results.map((x,i)=>{
   let cls=x.verdict==="POSSIBLY_SUPPORTED"?"supported":x.verdict==="POSSIBLE_CONTRADICTION"?"contradiction":"insufficient";
   return `<article class="result"><div class="topline"><strong>${i+1}. ${escapeHtml(x.claim)}</strong><span class="tag ${cls}">${x.verdict.replaceAll("_"," ")}</span></div><p>${escapeHtml(x.explanation)}</p><small>Overlap score: ${x.confidence}</small>${x.evidence?`<div class="evidence"><strong>Best matching passage</strong><br>${escapeHtml(x.evidence)}</div>`:""}</article>`;
  }).join("")||"<p>No claims could be extracted.</p>";
  $("limitations").innerHTML="<strong>Limitations</strong><ul>"+data.limitations.map(x=>`<li>${escapeHtml(x)}</li>`).join("")+"</ul>";
  $("status").textContent="Analysis complete.";
 }catch(e){$("status").textContent="Error: "+e.message;}
 finally{$("run").disabled=false;}
};
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
$("download").onclick=()=>{if(!lastReport)return;const blob=new Blob([JSON.stringify(lastReport,null,2)],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="truthlens-report.json";a.click();URL.revokeObjectURL(a.href);};
</script></body></html>"""
