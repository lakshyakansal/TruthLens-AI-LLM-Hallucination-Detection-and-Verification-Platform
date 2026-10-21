# TruthLens AI v2 — Render-ready LLM Hallucination Screening

TruthLens AI screens LLM-generated answers against user-supplied reference passages. It splits answers into sentence-level claims, retrieves the closest evidence passage, and can optionally use a trained Hugging Face Natural Language Inference (NLI) model to estimate whether evidence supports or contradicts each claim.

> **Important:** TruthLens AI is a screening aid, not a truth oracle. Its verdicts depend on the quality of the supplied evidence and the selected model. Do not use it as the sole basis for medical, legal, financial, or other high-impact decisions.

## Features

- Single-service web app and API designed for Render.
- Interactive dashboard at `/` for entering answers and evidence.
- Claim-level verdicts and evidence snippets.
- Optional Hugging Face NLI model for entailment, contradiction, or insufficient-evidence estimates.
- JSON report download from the dashboard.
- Health endpoint reports whether NLI is enabled.
- Swagger API documentation at `/docs`.
- Lightweight lexical fallback when NLI is disabled.

## Deploy on Render

### Option A: Deploy with a Render Blueprint (recommended)

1. Download and extract `truthlens_ai_render.zip`.
2. Upload the files from the extracted folder to the **root** of a GitHub repository (for example, `truthlens-ai`).
3. In Render, choose **New + → Blueprint** and connect the repository.
4. Render detects `render.yaml`. Review the configuration and create the service.
5. Wait for the deployment to finish, then open the public service URL.

### Option B: Create a Web Service manually

Use these settings:

- **Runtime:** Python
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Health check path:** `/health`

The baseline mode does not require API keys or environment variables.

## Enable model-based NLI verification (optional)

The default deployment is intentionally lightweight. By default, `ENABLE_NLI` is false, so the app uses a lexical-overlap baseline and will **not** claim that a trained NLI model is active.

To enable NLI, open your Render service → **Environment** and add:

| Key | Value |
|---|---|
| `ENABLE_NLI` | `true` |
| `NLI_MODEL_NAME` | `cross-encoder/nli-deberta-v3-small` |
| `NLI_THRESHOLD` | `0.65` |

The supplied `requirements.txt` includes Transformers and PyTorch for the optional model-backed mode. Redeploy after changing environment variables or dependencies.

**Resource warning:** PyTorch, Transformers, and the model weights are large. A free Render instance may run out of RAM or disk, take a long time to start, or fail to download the model. If that happens, return to `ENABLE_NLI=false`, or use a suitable paid instance or a hosted inference provider. Check `/health`; it reports `nli_enabled: true` only when NLI is configured as enabled. If the model fails to load, analysis requests return an error rather than silently pretending that NLI is working.

## Routes

- `/` — interactive TruthLens dashboard
- `/health` — service status and NLI configuration
- `/docs` — interactive API documentation
- `POST /analyze` — analyze claims against evidence

## API example

Send a `POST` request to `/analyze` with JSON:

```json
{
  "answer": "Water freezes at 0 degrees Celsius. The Sun revolves around Earth.",
  "evidence": [
    "Pure water freezes at 0 degrees Celsius under standard atmospheric pressure.",
    "Earth revolves around the Sun."
  ],
  "similarity_threshold": 0.35
}
```

Example using `curl`:

```bash
curl -X POST "https://YOUR-RENDER-SERVICE.onrender.com/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "answer": "Water freezes at 0 degrees Celsius.",
    "evidence": [
      "Pure water freezes at 0 degrees Celsius under standard atmospheric pressure."
    ]
  }'
```

Replace the URL with your actual Render service URL.

## How it works

1. Splits the generated response into sentence-level candidate claims.
2. Finds the supplied evidence passage with the highest lexical overlap.
3. If NLI is enabled, sends the best passage as the premise and the claim as the hypothesis to the NLI model.
4. Returns claim-level verdicts, confidence/model scores, the selected evidence, method, and limitations.
5. If NLI is disabled, returns conservative labels such as `POSSIBLY_SUPPORTED` or `INSUFFICIENT_EVIDENCE`. The lexical fallback does not establish factual truth.

## Verdict meanings

- `SUPPORTED` — the NLI model estimates that the evidence entails the claim.
- `CONTRADICTED` — the NLI model estimates that the evidence contradicts the claim.
- `INSUFFICIENT_EVIDENCE` — the evidence does not clearly support or contradict the claim, or no evidence is available.
- `POSSIBLY_SUPPORTED` — lexical overlap was found in fallback mode, but entailment was not verified.
- `NEEDS_REVIEW` — the NLI output could not be interpreted safely.

A contradicted claim is contradicted by the **supplied evidence**; this is not an independent web fact-check.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ENABLE_NLI` | `false` | Enable trained NLI inference when set to `true` |
| `NLI_MODEL_NAME` | `cross-encoder/nli-deberta-v3-small` | Hugging Face model identifier |
| `NLI_THRESHOLD` | `0.65` | Minimum model score used for supported/contradicted verdicts |
| `MAX_CLAIMS` | `40` | Maximum sentence-level claims processed per request |

## Current limitations

- Retrieval still uses lexical overlap and may miss paraphrases or relevant evidence with different wording.
- Sentence splitting is not full atomic claim extraction; a sentence can contain multiple claims.
- The NLI model estimates textual entailment; it does not independently establish real-world truth.
- Multi-passage reasoning, calculations, dates, temporal facts, and external source verification are not fully implemented.
- Model scores are not calibrated probabilities unless evaluated and calibrated on a representative labeled dataset.
- Human review is required for important decisions.

## Recommended next improvements

1. Replace lexical retrieval with Sentence Transformers embeddings and FAISS.
2. Add PDF upload, chunking, and page-level source citations.
3. Aggregate evidence from multiple passages for claims needing context.
4. Evaluate on a labeled benchmark; report precision, recall, F1, confusion matrix, false-positive rate, calibration, and latency.
5. Add persistent analysis history and authentication if needed.
6. Consider a hosted inference endpoint if Render resource limits prevent local NLI model execution.

<div align="center">

Developed by Lakshya Kansal

</div>
