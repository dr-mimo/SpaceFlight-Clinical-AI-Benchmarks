# Benchmarking Edge-Native Large Language Models for Autonomous Deep Space Medicine

**Status: Draft v0.2 — methodology paper. Results sections are templates to be filled in once the pipeline has been run against real model endpoints; see "How to complete this report" at the end.**

**Relationship to NASA/Google work:** this is an independent project, not affiliated with, produced by, or endorsed by NASA or Google. Where it references real named efforts (NASA's Human Research Program, ExMC, EIMO; Google/NASA's Crew Medical Officer Digital Assistant, "CMO-DA") it cites them accurately and at the general-public-information level — it does not reproduce non-public material from any of them, and it makes no claim that any of those programs use, endorse, or were involved in producing this benchmark.

## Abstract

Placeholder — write 150–200 words once real results exist: what was tested, on how many scenarios, headline finding, and the main limitation. Do not draft this until Sections 4–5 have real numbers, since abstracts written before results tend to overclaim.

## 1. Motivation

Missions beyond low Earth orbit will operate with communication delays that make real-time ground medical consultation impossible during acute events — up to roughly 20–24 minutes one-way at maximum Mars–Earth separation. NASA's Human Research Program (HRP) has publicly identified progressively Earth-independent medical operations (EIMO) as an open, high-priority capability gap for exploration-class missions, and the Exploration Medical Capability (ExMC) element has published conceptual work on a clinical decision support system (CDSS) for this purpose.

In August 2025, NASA and Google Public Sector jointly announced a related, real, named effort: the **Crew Medical Officer Digital Assistant (CMO-DA)**, a proof-of-concept CDSS built on Google Cloud's Vertex AI, evaluated using an Objective Structured Clinical Examination (OSCE)-style framework across three scenarios (ankle injury, flank pain, ear pain), with reported diagnostic-accuracy judgments of 88%, 80%, and 74% respectively from a reviewing panel that included physicians and an astronaut ([Google Cloud Blog, Aug 2025](https://cloud.google.com/blog/topics/public-sector/how-google-and-nasa-are-testing-ai-for-medical-care-in-space)).

This project is **independent of and has no affiliation with CMO-DA.** It exists to ask a narrower, openly reproducible question in the same problem space: **when given a synthetic acute clinical scenario relevant to spaceflight, how do current large language models — both cloud-hosted and fully local/disconnected — reason through diagnosis, resource-constrained treatment, and escalation, and how does that reasoning change when the model is told a ground consult is not available in time?**

This paper does not claim to answer whether any model is "safe" for real autonomous medical use, and does not claim any relationship to CMO-DA's actual architecture, training data, or results beyond citing it as prior public work in the same space. It reports a benchmark methodology and results against a synthetic, clinician-reviewed (but not peer-reviewed or NASA-validated) dataset.

## 2. Related context (public sources only)

This project draws only on **publicly available** material, cited at the level of general risk categories and named public programs rather than non-public document text:

- NASA Human Research Program (HRP) public Evidence Books and Human Research Roadmap; the Exploration Medical Capability (ExMC) element's public work on Earth-Independent Medical Operations (EIMO); NASA-STD-3001 Volume 1 ("Crew Health").
- The public NASA–Google CMO-DA collaboration, cited above, as evidence this problem space has active real-world interest — not as a technical reference for this repo's architecture.
- General hyperbaric and aerospace medicine literature on decompression sickness, blast/overpressure injury, and cardiac arrhythmia management (not spaceflight-specific in origin, but medically transferable, and reviewed by this project's clinical reviewer for spaceflight-specific adaptations — see `data/scenarios/*.json` `source_notes`).

No export-controlled, ITAR-restricted, or non-public NASA or Google data was used. This project has not been reviewed by, submitted to, or endorsed by NASA or Google, and should not be described as such.

## 3. Methodology

### 3.1 Dataset

`data/scenarios/` contains synthetic clinical vignettes across categories including decompression sickness, blast/overpressure trauma, cardiac events in microgravity, and radiation exposure triage (see `data/scenarios/schema.md`). Each scenario specifies:

- Mission context (vehicle, phase, crew size, one-way communication delay, ground-contact availability, and an explicit list of available medical resources)
- A presenting complaint and a short vitals timeline
- A weighted rubric of expected reasoning steps, tagged by category (triage accuracy, resource-appropriateness, autonomy-appropriateness, safety floor, escalation/documentation)
- A list of explicitly unsafe actions that trigger a scoring penalty if recommended

**Current dataset size:** 7 scenarios across 5 categories as of v0.3 (2 decompression sickness, 2 blast/overpressure trauma, 1 cardiac event in microgravity — all clinically reviewed; 1 acute radiation sickness, 1 toxic atmosphere exposure — both drafted but explicitly unreviewed pending clinician sign-off). See `data/scenarios/*.json` for the current count before publishing, since this is expected to grow. This is still a seed set, not a complete benchmark.

### 3.2 Models and backends evaluated

This benchmark's `ModelClient` (`src/model_client.py`) supports three backends, so the same scenarios and scoring can run against cloud and fully-disconnected models alike:

| Backend | What it represents | Example models |
|---|---|---|
| `vertex` | Cloud-hosted inference (approximates a connected/near-Earth scenario, or a ground-based second opinion) | `gemini-1.5-pro`, `gemini-1.5-flash` (check current Vertex AI model IDs before running — they change over time), or a custom Model Garden endpoint (e.g. a fine-tuned/deployed MedGemma) |
| `edge` | Fully local, no-outbound-network inference (approximates deep-space disconnection) | Any model served locally via an Ollama-compatible API, e.g. `llama3.1:8b`, or a local LiteRT/ONNX container |
| `mock` | Deterministic offline synthetic responses — **CI/testing only, never a real result** | `mock-model-a`, `mock-model-b` |

List the exact model IDs/versions and the backend used once you run the pipeline, e.g.:

| Model | Backend | Notes |
|---|---|---|
| *(fill in)* | vertex / edge / mock | fill in version/date, and for edge mode, the hardware it ran on |

Be precise about model version strings, backend, and the date you ran the evaluation — model behavior changes over time, edge-mode hardware affects latency (though not, in principle, output quality), and a benchmark that doesn't pin all three becomes misleading within months.

**Note on "edge-native":** this project's `edge` backend means "inference against a locally-hosted model with no outbound network call" — a software approximation of disconnection for benchmarking purposes. It does not reproduce actual flight hardware constraints (radiation-hardened compute, power/thermal budgets, qualified flight software). Do not cite a strong `edge`-backend result as evidence a model is qualified for real flight hardware.

### 3.3 Prompting

All models receive the same system instruction (see `src/eval_pipeline.py::SYSTEM_INSTRUCTION`) framing them as an onboard clinical decision-support assistant, instructed to reason step by step and to only use resources explicitly listed as available. Temperature and max tokens are held constant across models (see `src/vertex_client.py`) for comparability; note here if you deviate.

### 3.4 Scoring

Scoring formula, as specified by clinical review (Flight Surgeon / CMO Domain Lead, August 2026):

```
S_fit = (0.45 * A_dx + 0.55 * R_act) * S_safe
```

`A_dx` and `R_act` are the weighted fraction of matched rubric criteria in the `diagnosis` and `action` buckets respectively. `S_safe` is a **hard gate** (0 or 1, not additive): triggering any scenario's listed unsafe/contraindicated action zeroes the entire scenario score, regardless of diagnostic quality. This reflects a specific clinical judgment — that a correct diagnosis paired with a contraindicated intervention (e.g. PPV before decompressing a tension pneumothorax, nitrous oxide in suspected DCS, unrestrained defibrillation in microgravity) is not a partial success and should not be scored as one.

Two implementations are provided (`src/scoring.py`):

1. **Keyword heuristic** — fast, fully local, fully auditable, but crude; likely to both over- and under-credit responses on nuanced reasoning, and likely to both over- and under-trigger the safety gate on paraphrased unsafe recommendations.
2. **LLM-as-judge** — a second model call scores each response against the rubric and unsafe-action list; more sensitive to nuanced or paraphrased reasoning but introduces its own bias and should be spot-checked against human review, not treated as ground truth. Given that the safety gate is a hard zero, false positives/negatives here have an outsized effect on headline numbers — err toward the LLM-judge mode plus human spot-checking for anything published.

**State clearly in the final report which mode produced the headline numbers**, and report inter-method agreement (including specifically how often the two modes disagree on whether the safety gate was triggered) if you ran both.

### 3.5 Human review

All scenarios shipped as of v0.2 have been reviewed and signed off by a Flight Surgeon / CMO Domain Lead (see each scenario's `source_notes` for reviewer and date). Before publishing headline claims, additionally have at least one clinician review a random sample of *scored responses* (not just the rubrics) — the rubric can be clinically sound while the automated scorer still mis-applies it to a specific model response — and report agreement/disagreement rates here, with particular attention to safety-gate false positives/negatives given how much a single gate decision now swings the final score.

## 4. Results

**This section is a template. Do not fill in numbers you have not actually generated by running `src/eval_pipeline.py` against real Vertex AI model calls.** Replace the tables below with your actual output from `results/summary_*.md` plus the deeper per-category breakdowns.

### 4.1 Overall scores by model

| Model | Scenarios evaluated | Mean normalized score | Unsafe actions triggered |
|---|---|---|---|
| *(fill in)* | | | |

### 4.2 Scores by category

| Model | Decompression sickness | Blast/overpressure trauma | ... |
|---|---|---|---|
| *(fill in)* | | | |

### 4.3 Qualitative observations

*(fill in with specific, cited examples from actual transcripts in `results/responses_*.json` — e.g. "Model X's response to `dcs_002` correctly reasoned about the 15-minute comms delay by stating X, while Model Y's response assumed ground contact was available despite the prompt stating otherwise." Always link back to the scenario id and quote/paraphrase faithfully from your own generated output — never invent example outputs.)*

## 5. Comparison to NASA baseline expectations

This section should compare each model's rubric performance against the reasoning NASA's own public guidance would suggest (e.g. "start O2 immediately for suspected DCS regardless of chamber availability" per general aerospace medicine guidance) — **not** against any NASA-published model output, since NASA has not published LLM benchmark baselines for this task as of this writing. Be explicit that "NASA baseline expectations" here means *the rubric authors' best-effort synthesis of public NASA/aerospace-medicine guidance*, not an official NASA standard, and flag this prominently to avoid readers mistaking it for an official evaluation.

## 6. Limitations

- **Synthetic dataset, small N.** Results generalize only weakly; this is a seed benchmark, not a validated one.
- **No real clinical validation.** Rubrics were not (as of v0.1) reviewed by a licensed clinician — see `source_notes` in each scenario file for review status.
- **Scoring bias.** Both keyword and LLM-judge scoring have known failure modes (surface pattern matching vs. judge-model bias); treat scores as directional, not authoritative.
- **Prompt sensitivity.** LLM behavior can vary meaningfully with prompt phrasing; a single system instruction and prompt per scenario likely understates variance.
- **Not a certification.** No model evaluated here should be considered validated or approved for real autonomous medical use based on this benchmark.

## 7. Future work

- Expand scenario coverage (target: 10+ scenarios per category, clinician-reviewed).
- Add inter-rater reliability analysis between keyword scoring, LLM-judge scoring, and human clinician scoring.
- Add adversarial/edge-case scenarios (ambiguous presentations, conflicting vitals, equipment failure mid-treatment).
- Track model performance over time as new model versions ship, with pinned version/date per run.

## How to complete this report

1. Pick a backend and get real numbers:
   - **Free/offline sanity check:** `python src/eval_pipeline.py --backend mock --models mock-model-a,mock-model-b --out results/` (pipeline plumbing only, not real results — do not put these in Section 4).
   - **Edge/local:** install [Ollama](https://ollama.com), `ollama pull <model>`, then `python src/eval_pipeline.py --backend edge --models <model> --out results/`.
   - **Vertex/cloud:** get GCP/Vertex AI access, `export GOOGLE_CLOUD_PROJECT=...`, then `python src/eval_pipeline.py --backend vertex --models <ids> --out results/`.
2. Optionally add `--judge-model <id>` for LLM-as-judge scoring (queried via the same backend).
3. Fill in Sections 4–5 using only numbers/quotes from your own `results/` output, and label every result with the backend and exact model ID/version used.
4. Confirm the clinical review pass (Section 3.5) covers every scenario used, and get sign-off on the scored *responses*, not just the rubrics, before calling anything "benchmark-grade."
5. Update the abstract last.

---

*This document is an independent research artifact. It is not published, produced, or endorsed by NASA or Google, and should be presented as such wherever it is shared. Where it references CMO-DA or NASA HRP/ExMC/EIMO work, those references are to real public efforts, cited accurately — this project is not part of, and makes no claim to be part of, any of them.*
