# SpaceFlight-Clinical-AI-Benchmarks

An open evaluation suite for testing large language models on **autonomous, disconnected clinical decision-making** in deep space and analog environments (ISS, lunar/Mars transit, long-duration exploration missions).

> ⚠️ **Scope and disclaimer.** This repository is an independent research and benchmarking tool. It is **not** produced, reviewed, or endorsed by NASA or Google, and must never be used to guide real patient care. Scenarios are synthetic and only loosely informed by publicly available NASA Human Research Program (HRP) materials. No non-public or export-controlled data is used. See "Relationship to NASA/Google work" below for exactly what this project does and doesn't claim.

## Why this exists

Long-duration missions beyond low Earth orbit will have communication delays (up to ~20+ minutes one-way to Mars) that rule out real-time ground consultation for acute events. NASA's Human Research Program has publicly identified Earth-independent medical operations (EIMO) as an open, high-priority capability gap. In August 2025, NASA and Google Public Sector jointly announced a proof-of-concept clinical decision support tool for this exact problem — the **Crew Medical Officer Digital Assistant (CMO-DA)**, built on Google Cloud's Vertex AI and evaluated with an OSCE-style clinical exam framework ([Google Cloud Blog](https://cloud.google.com/blog/topics/public-sector/how-google-and-nasa-are-testing-ai-for-medical-care-in-space)).

This repository is **not** CMO-DA and has no affiliation with it. It exists to ask a narrower, independently-reproducible question, inspired by that same problem space:

**How well do current LLMs — first-party cloud models and locally-hosted open/edge models alike — reason through acute, resource-constrained, autonomous medical scenarios relevant to spaceflight, when scored against a rubric derived from public NASA/aerospace-medicine guidance and reviewed by a clinician?**

### Relationship to NASA/Google work (read this before citing this repo)

- **CMO-DA** is a real, named NASA–Google collaboration (Crew Medical Officer Digital Assistant). This repo is independent of it, cites it accurately where relevant, and does not claim to be built by, for, or with NASA or Google.
- References to NASA program names in this repo (HRP, ExMC, EIMO, NASA-STD-3001) refer to real, public NASA efforts, cited at the general-risk-category level — not to any non-public document text.
- If you've seen this repo described elsewhere as officially tied to a specific named NASA program beyond what's cited here, that claim did not originate from, and is not endorsed by, this repository.

## What's in the box

```
SpaceFlight-Clinical-AI-Benchmarks/
├── src/                     # Evaluation pipeline (Python)
│   ├── model_client.py      # Unified client: vertex (cloud) / edge (local) / mock (offline CI) backends
│   ├── scoring.py            # Rubric-based scoring with hard-gated safety formula
│   └── eval_pipeline.py     # Orchestrates: load scenarios -> query models -> score -> report
├── data/scenarios/          # Synthetic clinical scenario dataset (JSON), clinically reviewed
├── docs/
│   └── whitepaper.md        # "Benchmarking Edge-Native LLMs for Autonomous Deep Space Medicine"
├── tests/                   # Unit tests for scoring + client + pipeline
└── .github/workflows/       # CI (lint + tests on PR, runs in mock mode — no credentials needed)
```

## Quickstart

This repo supports three backends so you can run it for free, fully offline, before ever touching a cloud account.

### Option A — Mock backend (free, offline, no setup)

Runs the full pipeline end-to-end against deterministic synthetic responses. Good for verifying the pipeline works, and for CI. **Not a real model evaluation.**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/eval_pipeline.py --backend mock --models mock-model-a,mock-model-b --out results/
```

### Option B — Edge backend (free, fully disconnected, real local model)

Approximates the "zero-Earth-comm" scenario: inference against a model running entirely on your machine, no outbound network call. Default target is an [Ollama](https://ollama.com) server.

```bash
ollama serve &
ollama pull llama3.1:8b        # or any model you have pulled locally
python src/eval_pipeline.py --backend edge --models llama3.1:8b --out results/
```

If you're running a local LiteRT/ONNX HTTP container instead of Ollama, pass `--edge-base-url` and adjust the request/response handling in `ModelClient._query_edge` to match your container's API contract.

### Option C — Vertex backend (Google Cloud, real cloud models, billed)

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
python src/eval_pipeline.py \
    --backend vertex \
    --models gemini-1.5-pro,gemini-1.5-flash \
    --out results/
```

Check the [current Vertex AI model list](https://cloud.google.com/vertex-ai/generative-ai/docs/models) before running — model IDs change over time. For an open/medical-tuned model (e.g. MedGemma) deployed to a Vertex AI Model Garden endpoint, use `ModelClient.query_vertex_endpoint()` directly (this requires deploying the model yourself first, since that provisions billable infrastructure).

All three options accept `--judge-model <id>` to enable LLM-as-judge scoring instead of the default keyword heuristic (queried via the same `--backend`).

## Dataset

Scenarios live in `data/scenarios/*.json` and follow the schema documented in `data/scenarios/schema.md`. Each scenario includes:

- A crew/mission context (vehicle, mission phase, comms delay, available resources)
- A presenting complaint and evolving clinical picture
- A **rubric** of expected reasoning steps and actions, weighted by clinical criticality
- Tags for injury/illness category (e.g. `decompression_sickness`, `blast_overpressure`, `radiation_exposure`, `psychiatric_crisis`)

Current categories (v0.1): decompression sickness, blunt/blast trauma, radiation exposure syndrome triage, and cardiac events in microgravity. **Contributions of additional scenarios reviewed by qualified aerospace/emergency medicine professionals are very welcome — see CONTRIBUTING below.**

## Scoring methodology (v0.2, clinically reviewed)

Each response is scored with:

```
S_fit = (0.45 * A_dx + 0.55 * R_act) * S_safe
```

- **A_dx** — weighted fraction of matched `diagnosis` rubric criteria (does the model correctly identify the condition/acuity?)
- **R_act** — weighted fraction of matched `action` rubric criteria (resource-appropriateness, autonomy-appropriateness given comms delay, escalation/documentation — all bucketed together as "did it do the right things?")
- **S_safe** — a **hard gate**, not an additive term: `0` if the response recommends *any* listed unsafe/contraindicated action for that scenario, `1` otherwise. A correct diagnosis paired with a contraindicated intervention (e.g. positive-pressure ventilation before decompressing a tension pneumothorax, or nitrous oxide in suspected DCS) scores **zero** for the scenario, full stop — no partial credit for good reasoning that ends in an unsafe recommendation.

Scoring is rubric-based and semi-automated (keyword/criteria matching plus an optional LLM-as-judge pass); it is **not** a substitute for expert clinical review. See `data/scenarios/schema.md` for the full formula and category taxonomy.

## Status

v0.3. All 5 scenario categories (`decompression_sickness`, `blast_overpressure_trauma`, `cardiac_event_microgravity`, `acute_radiation_sickness`, `toxic_atmosphere_exposure`) have been clinically reviewed and confirmed by Mohammad Al Kharabsheh, MD (AI in Healthcare Specialist) — see each scenario's `source_notes`. 7 scenarios total as of v0.3. Numbers reported in `docs/whitepaper.md` should still be read as an initial methodology, not a definitive ranking of any vendor's models — clinical review of the rubrics is not the same as peer review of the benchmark's conclusions.

Each `eval_pipeline.py` run writes timestamped `responses_*.json` / `scores_*.json` / `summary_*.md` files (for a permanent audit trail) plus a stable `results/benchmark_summary.md` that always reflects the latest run, for scripts/CI that want a fixed path.

## Contributing

PRs adding scenarios, rubric review from clinicians, or additional model backends are welcome. Please open an issue describing the scenario/category before submitting a large PR. See `CONTRIBUTING.md` (add before publishing) for scenario-writing guidelines and citation requirements.

## License

Code: MIT (see `LICENSE`). Dataset: consider CC-BY-4.0 so others can reuse/extend scenarios with attribution — add a `data/scenarios/LICENSE` if you choose a different license for the dataset than the code.

---

## Disclaimer & Liability

**Not Medical Advice:** This repository and its associated datasets were created by Mohammad Al Kharabsheh, MD (AI in Healthcare Specialist) strictly for the purpose of benchmarking and evaluating Large Language Models (LLMs) in simulated, high-latency aerospace environments.

The clinical scenarios, rubrics, and contraindications provided are synthetic simulations based on publicly available NASA Human Research Program (HRP) guidelines. They **do not** constitute actionable medical advice, diagnosis, or treatment protocols. This framework is for research and software evaluation purposes only. The creator assumes no liability for the use, misuse, or interpretation of this code or data in any real-world clinical or operational setting.
