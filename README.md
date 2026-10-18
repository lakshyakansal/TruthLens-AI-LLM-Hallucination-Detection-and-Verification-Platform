# TruthLens AI — Render-ready LLM Hallucination Screening

A single-service, Render-deployable web app and API for screening LLM answers against user-supplied reference passages.

## Deploy on Render

### Option A: Deploy from GitHub (recommended)

1. Download and extract `truthlens_ai_render.zip`.
2. Create a new GitHub repository, for example `truthlens-ai`.
3. Upload all files from the extracted folder to the repository root.
4. In Render, choose **New + → Blueprint** and connect the repository.
5. Render detects `render.yaml`; review and create the service.
6. Wait for deployment, then open the service URL.

### Option B: Create a Web Service manually

- Runtime: **Python**
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

No environment variables or API keys are required for this baseline version.

## Routes

- `/` — interactive dashboard
- `/health` — health check
- `/docs` — interactive API documentation
- `POST /analyze` — analyze claims against evidence

## API example

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

## How it works

1. Splits the response into sentence-level candidate claims.
2. Finds the evidence passage with the highest word overlap.
3. Flags likely overlap and negation mismatches for human review.
4. Returns a JSON report with claim-level results and limitations.

## Important limitations

This is a **deployment-ready starter**, not a validated hallucination detector. To keep the free Render service lightweight and avoid large model downloads, it uses a lexical baseline rather than a trained NLI model. Word overlap is not proof of entailment, and negation heuristics can produce false positives and false negatives. The app must not be used as the sole basis for medical, legal, financial, or other high-impact decisions.

Recommended next step: add Sentence Transformers + an NLI model or a provider API, then evaluate on a labeled benchmark before describing accuracy claims.
