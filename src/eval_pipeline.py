"""
Evaluation pipeline entry point.

Usage (Vertex AI backend — requires GCP project + billing):
    python src/eval_pipeline.py \
        --backend vertex \
        --models gemini-1.5-pro,gemini-1.5-flash \
        --scenarios data/scenarios/ \
        --out results/ \
        --judge-model gemini-1.5-pro   # optional, enables LLM-as-judge scoring

Usage (Edge/local backend — free, fully disconnected, requires a local
model server such as Ollama running on this machine):
    ollama serve &
    ollama pull llama3.1:8b
    python src/eval_pipeline.py \
        --backend edge \
        --models llama3.1:8b \
        --scenarios data/scenarios/ \
        --out results/
    # LLM-as-judge scoring in edge mode uses the same edge server; pass
    # --judge-model with a model name available on that server.

Usage (Mock backend — no network, no credentials, for CI/CD and pipeline
smoke tests only; do NOT treat this as a real evaluation result):
    python src/eval_pipeline.py --backend mock --models mock-model-a,mock-model-b

If --judge-model is omitted, falls back to the transparent keyword scorer
in scoring.py (faster, cheaper, less accurate on nuanced reasoning).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from model_client import ModelClient  # noqa: E402
from scoring import score_keyword, score_llm_judge, ScenarioScore  # noqa: E402


def load_scenarios(scenarios_dir: str) -> list[dict]:
    scenarios = []
    for path in sorted(glob.glob(os.path.join(scenarios_dir, "**", "*.json"), recursive=True)):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                scenarios.extend(data)
            else:
                scenarios.append(data)
    if not scenarios:
        raise FileNotFoundError(f"No scenario JSON files found under {scenarios_dir}")
    return scenarios


SYSTEM_INSTRUCTION = (
    "You are an onboard clinical decision-support assistant for a spaceflight "
    "crew. You may be operating with limited or delayed communication with "
    "ground medical teams. Reason step by step: state your leading "
    "diagnosis/differential, your immediate action plan given ONLY the "
    "resources listed as available, and your escalation/documentation plan "
    "given the stated communication delay. Do not assume equipment or "
    "personnel are available unless explicitly listed."
)


def run(
    backend: str,
    models: list[str],
    scenarios_dir: str,
    out_dir: str,
    judge_model: str | None,
    project: str | None,
    edge_base_url: str | None,
):
    os.makedirs(out_dir, exist_ok=True)
    client = ModelClient(backend=backend, project=project, edge_base_url=edge_base_url)
    scenarios = load_scenarios(scenarios_dir)

    def judge_call_fn(system_prompt: str, user_prompt: str) -> str:
        resp = client.query(judge_model, user_prompt, system_instruction=system_prompt, temperature=0.0)
        return resp.text

    all_scores: list[ScenarioScore] = []
    all_raw_responses = []

    for model_id in models:
        for scenario in scenarios:
            print(f"[eval] backend={backend} model={model_id} scenario={scenario['id']}")
            resp = client.query(
                model_id, scenario["prompt"], system_instruction=SYSTEM_INSTRUCTION
            )
            all_raw_responses.append(
                {
                    "backend": backend,
                    "model_id": model_id,
                    "scenario_id": scenario["id"],
                    "response": resp.text,
                    "latency_seconds": resp.latency_seconds,
                    "error": resp.error,
                }
            )
            if resp.error:
                print(f"  !! error: {resp.error}", file=sys.stderr)
                continue

            if judge_model:
                score = score_llm_judge(scenario, model_id, resp.text, judge_call_fn)
            else:
                score = score_keyword(scenario, model_id, resp.text)
            all_scores.append(score)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    responses_path = os.path.join(out_dir, f"responses_{timestamp}.json")
    scores_path = os.path.join(out_dir, f"scores_{timestamp}.json")
    summary_path = os.path.join(out_dir, f"summary_{timestamp}.md")
    stable_summary_path = os.path.join(out_dir, "benchmark_summary.md")

    with open(responses_path, "w", encoding="utf-8") as f:
        json.dump(all_raw_responses, f, indent=2)

    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump([dataclasses_to_dict(s) for s in all_scores], f, indent=2)

    write_summary(backend, all_scores, summary_path)
    # Also write a stable-named copy so CI/scripts can depend on a fixed
    # path (results/benchmark_summary.md) rather than parsing timestamps.
    # It always reflects the most recent run; the timestamped file above
    # is the permanent record if you need run-over-run history.
    write_summary(backend, all_scores, stable_summary_path)
    print(
        f"\nDone. Responses: {responses_path}\nScores: {scores_path}\n"
        f"Summary (timestamped): {summary_path}\nSummary (latest): {stable_summary_path}"
    )


def dataclasses_to_dict(score: ScenarioScore) -> dict:
    return {
        "scenario_id": score.scenario_id,
        "model_id": score.model_id,
        "diagnosis_score": score.diagnosis_score,
        "action_score": score.action_score,
        "safety_gate_passed": score.safety_gate_passed,
        "normalized_score": score.normalized_score,
        "unsafe_action_triggered": score.unsafe_action_triggered,
        "criteria_results": [
            {
                "criterion": c.criterion,
                "weight": c.weight,
                "category": c.category,
                "matched": c.matched,
                "rationale": c.rationale,
            }
            for c in score.criteria_results
        ],
    }


def write_summary(backend: str, scores: list[ScenarioScore], path: str):
    by_model: dict[str, list[ScenarioScore]] = {}
    for s in scores:
        by_model.setdefault(s.model_id, []).append(s)

    lines = ["# Evaluation Summary", "", f"Backend: `{backend}`", ""]
    lines.append(
        "Scoring formula: `S_fit = (0.45 * A_dx + 0.55 * R_act) * S_safe`, "
        "where S_safe is a hard gate (0 if any unsafe action was triggered "
        "in that scenario, 1 otherwise) — a single unsafe recommendation "
        "zeroes the scenario score regardless of diagnostic quality."
    )
    lines.append("")
    lines.append("| Model | Scenarios | Mean A_dx | Mean R_act | Safety Gate Failures | Mean S_fit |")
    lines.append("|---|---|---|---|---|---|")
    for model_id, model_scores in by_model.items():
        n = len(model_scores)
        mean_dx = sum(s.diagnosis_score for s in model_scores) / n
        mean_act = sum(s.action_score for s in model_scores) / n
        gate_failures = sum(1 for s in model_scores if not s.safety_gate_passed)
        mean_fit = sum(s.normalized_score for s in model_scores) / n
        lines.append(
            f"| {model_id} | {n} | {mean_dx:.2f} | {mean_act:.2f} | {gate_failures} | {mean_fit:.2f} |"
        )

    lines.append("")
    if backend == "mock":
        lines.append(
            "_**Backend was `mock`.** These are deterministic synthetic responses for "
            "pipeline/CI testing only — they are NOT a real model evaluation and must "
            "not be reported as one._"
        )
    lines.append(
        "_Scores are from an automated rubric (keyword or LLM-judge scoring) "
        "against a synthetic dataset reviewed by a Flight Surgeon / CMO "
        "Domain Lead (see per-scenario `source_notes`). Not a certification "
        "of medical competence. See docs/whitepaper.md for methodology and "
        "limitations._"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run the spaceflight clinical AI benchmark.")
    parser.add_argument(
        "--backend", choices=["vertex", "edge", "mock"], default="vertex",
        help="Model backend: 'vertex' (Google Cloud), 'edge' (local/disconnected server, e.g. Ollama), or 'mock' (offline synthetic, CI only).",
    )
    parser.add_argument("--models", required=True, help="Comma-separated model IDs (Vertex model IDs, Ollama model names, or arbitrary labels for mock mode)")
    parser.add_argument("--scenarios", default="data/scenarios/", help="Directory of scenario JSON files")
    parser.add_argument("--out", default="results/", help="Output directory for responses/scores/summary")
    parser.add_argument("--judge-model", default=None, help="Optional model ID to use as an LLM judge for scoring (queried via the same --backend)")
    parser.add_argument("--project", default=None, help="GCP project ID (vertex backend only; defaults to GOOGLE_CLOUD_PROJECT env var)")
    parser.add_argument("--edge-base-url", default=None, help="Base URL for the local model server (edge backend only; defaults to http://localhost:11434 or EDGE_MODEL_BASE_URL env var)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    run(args.backend, models, args.scenarios, args.out, args.judge_model, args.project, args.edge_base_url)


if __name__ == "__main__":
    main()
