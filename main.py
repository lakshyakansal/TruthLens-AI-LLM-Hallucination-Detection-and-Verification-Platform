import os
import re
import json
from typing import List, Dict, Any, Optional
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="TruthLens AI",
    description="Evidence-grounded LLM hallucination detection with optional NLI",
    version="2.0.0",
)

# Model loading is lazy so the service can boot quickly on Render.
# Set ENABLE_NLI=true to enable a real Hugging Face NLI classifier.
NLI_MODEL_NAME = os.getenv("NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-small")
ENABLE_NLI = os.getenv("ENABLE_NLI", "false").lower() in {"1", "true", "yes"}
NLI_THRESHOLD = float(os.getenv("NLI_THRESHOLD", "0.65"))
MAX_CLAIMS = int(os.getenv("MAX_CLAIMS", "40"))


def split_claims(text: str) -> List[str]:
    """Simple sentence-level claim candidates; not full atomic claim extraction."""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()][:MAX_CLAIMS]


def tokenize(text: str) -> set:
    return set(re.findall(r"\b[\w'-]+\b", text.lower()))


def lexical_score(claim: str, passage: str) -> float:
    c = tokenize(claim)
    p = tokenize(passage)
    return len(c & p) / max(1, len(c))


@lru_cache(maxsize=1)
def get_nli_pipeline():
    if not ENABLE_NLI:
        return None
    try:
        from transformers import pipeline
        return pipeline(
            "text-classification",
            model=NLI_MODEL_NAME,
            tokenizer=NLI_MODEL_NAME,
            top_k=None,
            truncation=True,
        )
    except Exception as exc:
        # Do not silently pretend the NLI model is active.
        raise RuntimeError(f"NLI model could not be loaded: {exc}") from exc


def normalize_nli_output(raw) -> Dict[str, float]:
    """Normalize common Transformers text-classification output shapes."""
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]
    if isinstance(raw, dict):
        raw = [raw]
    scores = {}
    for item in raw or []:
        label = str(item.get("label", "")).lower()
        scores[label] = float(item.get("score", 0.0))
    return scores


def find_score(scores: Dict[str, float], word: str) -> float:
    return max((v for k, v in scores.items() if word in k), default=0.0)


def nli_verdict(claim: str, evidence: str) -> Optional[Dict[str, Any]]:
    nli = get_nli_pipeline()
    if nli is None:
        return None
    # NLI premise = evidence, hypothesis = claim.
    raw = nli({"text": evidence, "text_pair": claim})
    scores = normalize_nli_output(raw)
    entail = find_score(scores, "entail")
    contradiction = find_score(scores, "contrad")
    neutral = find_score(scores, "neutral")

    # Labels vary by model; fail closed if the model doesn't return recognizable labels.
    if not any(("entail" in k or "contrad" in k or "neutral" in k) for k in scores):
        return {
            "verdict": "NEEDS_REVIEW",
            "confidence": 0.0,
            "explanation": "The selected NLI model returned labels this app could not interpret.",
            "scores": scores,
        }

    if entail >= NLI_THRESHOLD and entail > contradiction:
        verdict = "SUPPORTED"
        confidence = entail
        explanation = "The NLI model estimates that the supplied evidence entails this claim."
    elif contradiction >= NLI_THRESHOLD and contradiction > entail:
        verdict = "CONTRADICTED"
        confidence = contradiction
        explanation = "The NLI model estimates that the supplied evidence contradicts this claim."
    else:
        verdict = "INSUFFICIENT_EVIDENCE"
        confidence = max(neutral, 1.0 - max(entail, contradiction))
        explanation = "The evidence does not clearly entail or contradict this claim."
    return {
        "verdict": verdict,
        "confidence": round(float(confidence), 4),
        "explanation": explanation,
        "nli_scores": {k: round(v, 4) for k, v in scores.items()},
    }


def analyze(answer: str, evidence: List[str], threshold: float = 0.30) -> Dict[str, Any]:
    claims = split_claims(answer)
    passages = [re.sub(r"\s+", " ", p).strip() for p in evidence if p and p.strip()]
    output = []

    for claim in claims:
        if not passages:
            output.append({
                "claim": claim,
                "verdict": "INSUFFICIENT_EVIDENCE",
                "confidence": 0.0,
                "evidence": None,
                "explanation": "No reference evidence was provided. This is not proof that the claim is false.",
                "method": "no_evidence",
            })
            continue

        ranked = sorted(
            ((lexical_score(claim, passage), passage) for passage in passages),
            key=lambda x: x[0],
            reverse=True,
        )
        best_score, best_passage = ranked[0]

        try:
            nli = nli_verdict(claim, best_passage)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        if nli is not None:
            result = {
                "claim": claim,
                **nli,
                "evidence": best_passage,
                "retrieval_score": round(best_score, 4),
                "method": f"lexical_retrieval+nli:{NLI_MODEL_NAME}",
            }
        else:
            # Conservative fallback. It never labels a claim definitely supported/false.
            if best_score >= threshold:
                verdict = "POSSIBLY_SUPPORTED"
                explanation = (
                    "Relevant terms overlap with this passage, but no NLI model is enabled. "
                    "Word overlap cannot establish entailment or contradiction."
                )
            else:
                verdict = "INSUFFICIENT_EVIDENCE"
                explanation = (
                    "No sufficiently overlapping passage was found. This does not prove the claim is false."
                )
            result = {
                "claim": claim,
                "verdict": verdict,
                "confidence": round(best_score, 4),
                "evidence": best_passage,
                "retrieval_score": round(best_score, 4),
                "explanation": explanation,
                "method": "lexical_baseline",
            }
        output.append(result)

    counts = {}
    for item in output:
        counts[item["verdict"]] = counts.get(item["verdict"], 0) + 1

    total = len(output)
    evidence_found = sum(bool(item.get("evidence")) for item in output)
    return {
        "summary": {
            "total_claims": total,
            "verdict_counts": counts,
            "evidence_coverage": round(evidence_found / total, 4) if total else 0.0,
            "nli_enabled": ENABLE_NLI,
            "nli_model": NLI_MODEL_NAME if ENABLE_NLI else None,
        },
        "results": output,
        "limitations": [
            "NLI estimates whether evidence entails or contradicts a claim; it does not independently establish real-world truth.",
            "The current retrieval step uses lexical overlap and may miss paraphrases or relevant passages.",
            "Claims requiring multiple passages, arithmetic, time-sensitive facts, or external knowledge may need specialized verification.",
            "Confidence values are model scores, not calibrated probabilities unless separately calibrated on a representative benchmark.",
            "Human review is required for high-impact decisions.",
        ],
    }


class AnalyzeRequest(BaseModel):
    answer: str = Field(..., min_length=1, max_length=30000)
    evidence: List[str] = Field(default_factory=list, max_length=100)
    similarity_threshold: float = Field(default=0.30, ge=0.0, le=1.0)


@app.get("/health")
def health():
    return {"status": "ok", "service": "truthlens-ai", "nli_enabled": ENABLE_NLI}


@app.post("/analyze")
def analyze_endpoint(request: AnalyzeRequest):
    return analyze(request.answer, request.evidence, request.similarity_threshold)


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TruthLens AI</title>
<style>
:root{color-scheme:dark;--bg:#0a1020;--panel:#121c32;--line:#2a3958;--text:#eff3ff;--muted:#aab7d0;--accent:#9aaaff;--green:#80e0b9;--red:#ff9eac;--yellow:#ffd58a}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at top,#19294d,var(--bg) 60%);color:var(--text);font:16px/1.55 system-ui,sans-serif}.wrap{max-width:1100px;margin:auto;padding:34px 20px 60px}.brand{display:flex;gap:12px;align-items:center}.logo{height:44px;width:44px;border-radius:14px;background:#293b66;display:grid;place-items:center;font-size:24px}h1{font-size:clamp(2rem,5vw,3.3rem);line-height:1.08;letter-spacing:-.04em;margin:28px 0 12px}.grad{color:#a9b6ff}.muted,small{color:var(--muted)}.lead{max-width:750px;color:var(--muted)}.panel{background:#121c32ee;border:1px solid var(--line);border-radius:18px;padding:22px;margin-top:24px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.field{display:flex;flex-direction:column;gap:8px}label{font-weight:700}textarea{width:100%;min-height:185px;resize:vertical;background:#0b1428;border:1px solid #354666;border-radius:12px;color:var(--text);padding:13px;font:14px/1.6 ui-monospace,monospace}textarea:focus{outline:2px solid var(--accent)}.actions{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:18px}button{border:0;border-radius:11px;padding:12px 18px;font-weight:750;cursor:pointer;background:#a6b2ff;color:#10172c}button.secondary{background:#283958;color:var(--text)}button:disabled{opacity:.55}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.metric,.result{border:1px solid var(--line);background:#0d172b;border-radius:13px;padding:15px}.metric strong{display:block;font-size:1.65rem}.metric span{font-size:.84rem;color:var(--muted)}.result{margin-top:12px}.topline{display:flex;justify-content:space-between;gap:12px;align-items:start}.tag{font-size:.72rem;font-weight:850;padding:4px 8px;border-radius:8px;background:#273756}.supported{color:var(--green)}.contradicted,.possible_contradiction{color:var(--red)}.insufficient_evidence,.needs_review,.possibly_supported{color:var(--yellow)}.evidence{margin-top:12px;padding:10px 12px;background:#172442;border-left:3px solid #7f91ee;white-space:pre-wrap}.notice{font-size:.88rem;color:var(--muted);border-top:1px solid var(--line);margin-top:22px;padding-top:15px}.hidden{display:none}.error{color:var(--red)}@media(max-width:720px){.grid{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.panel{padding:16px}}
</style></head><body><main class="wrap">
<div class="brand"><div class="logo">⌕</div><div><strong>TRUTHLENS AI</strong><div class="muted" style="font-size:.78rem">LLM CLAIM VERIFICATION</div></div></div>
<h1>Check the answer.<br><span class="grad">Inspect the evidence.</span></h1>
<p class="lead">Analyze AI-generated claims against supplied reference passages. Enable a trained Natural Language Inference model to distinguish support, contradiction, and insufficient evidence.</p>
<section class="panel"><div class="grid">
<div class="field"><label for="answer">LLM-generated answer</label><textarea id="answer" placeholder="Paste the response to check..."></textarea><small>Each sentence is treated as a candidate claim.</small></div>
<div class="field"><label for="evidence">Reference evidence</label><textarea id="evidence" placeholder="Paste trusted passages, one per line..."></textarea><small>Use reliable source text relevant to the answer.</small></div></div>
<div class="actions"><button id="run">Analyze claims →</button><button id="demo" class="secondary">Load demo</button><span id="status" class="muted" role="status"></span></div></section>
<section id="output" class="panel hidden" aria-live="polite"><h2>Verification report</h2><div id="metrics" class="metrics"></div><div id="results"></div><div class="actions"><button id="download" class="secondary">Download JSON report</button></div><div id="limitations" class="notice"></div></section>
<p class="notice">A detector estimates evidence support; it is not a truth oracle. Unsupported claims are not automatically false. Review important results, especially for medical, legal, or financial use.</p>
</main><script>
let report=null;const $=id=>document.getElementById(id);
function esc(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
$("demo").onclick=()=>{$("answer").value="Water freezes at 0 degrees Celsius. The Sun revolves around Earth."; $("evidence").value="Pure water freezes at 0 degrees Celsius under standard atmospheric pressure.\\nEarth orbits the Sun.";};
$("run").onclick=async()=>{const answer=$("answer").value.trim(),evidence=$("evidence").value.split("\\n").map(s=>s.trim()).filter(Boolean);if(!answer){$("status").textContent="Enter an answer first.";return;} $("run").disabled=true;$("status").textContent="Analyzing…";try{const r=await fetch("/analyze",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({answer,evidence})});const data=await r.json();if(!r.ok)throw new Error(data.detail||"Request failed");report=data;$("output").classList.remove("hidden");const s=data.summary;const entries=[["Claims",s.total_claims],["Supported",s.verdict_counts.SUPPORTED||0],["Contradicted",s.verdict_counts.CONTRADICTED||0],["Insufficient evidence",s.verdict_counts.INSUFFICIENT_EVIDENCE||0]];$("metrics").innerHTML=entries.map(([k,v])=>`<div class="metric"><strong>${v}</strong><span>${k}</span></div>`).join("");$("results").innerHTML=data.results.map((x,i)=>`<article class="result"><div class="topline"><strong>${i+1}. ${esc(x.claim)}</strong><span class="tag ${x.verdict.toLowerCase()}">${esc(x.verdict.replaceAll("_"," "))}</span></div><p>${esc(x.explanation)}</p><small>Score: ${x.confidence} · Method: ${esc(x.method)}</small>${x.evidence?`<div class="evidence"><strong>Evidence</strong><br>${esc(x.evidence)}</div>`:""}</article>`).join("")||"<p>No claims extracted.</p>";$("limitations").innerHTML="<strong>Limitations</strong><ul>"+data.limitations.map(x=>`<li>${esc(x)}</li>`).join("")+"</ul>";$("status").textContent=s.nli_enabled?"Analysis complete · NLI enabled":"Analysis complete · lexical fallback";}catch(e){$("status").textContent="Error: "+e.message;}finally{$("run").disabled=false;}};
$("download").onclick=()=>{if(!report)return;const a=document.createElement("a"),u=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:"application/json"}));a.href=u;a.download="truthlens-report.json";a.click();URL.revokeObjectURL(u);};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML
