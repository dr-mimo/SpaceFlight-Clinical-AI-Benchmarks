"""
Unified model client with three interchangeable backends:

- "vertex": Google Vertex AI — first-party Gemini models, or a custom model
  (e.g. a fine-tuned/deployed MedGemma) served from a Vertex AI Model
  Garden endpoint.
- "edge":   A locally-hosted, disconnected model server — default target is
  an Ollama-compatible HTTP API (http://localhost:11434), which also covers
  most local LiteRT/ONNX runtime HTTP containers that expose a similar
  POST /api/generate-style interface. Used to approximate "zero-Earth-comm"
  inference: no outbound network call leaves the machine running this code.
- "mock":   Deterministic, offline synthetic responses. No network calls,
  no credentials, no local model server required. For unit tests / CI only
  — never treat mock-mode output as a real model evaluation result.

Important scope note: "edge mode" here means "inference against a
locally-hosted model with no outbound network calls." It is a software
approximation of a disconnected deep-space scenario for benchmarking
purposes — it does not reproduce actual flight hardware constraints
(radiation-hardened compute, power/thermal budgets, qualified flight
software, etc.), and this repo makes no claim that any model tested this
way is qualified for real flight hardware.
"""

from __future__ import annotations

import os
import time
import json
import hashlib
import dataclasses
from typing import Optional, Literal

import requests


@dataclasses.dataclass
class ModelResponse:
    model_id: str
    backend: str
    prompt: str
    text: str
    latency_seconds: float
    error: Optional[str] = None


# Deterministic mock responses keyed by scenario-relevant keyword so that
# pipeline plumbing and scoring logic can be exercised end-to-end in CI
# without hitting any network. These are NOT meant to resemble a good or
# bad model response — they exist purely to make mock mode return
# different, reproducible text for different prompts.
_MOCK_LIBRARY = {
    "decompression": (
        "Mock response: presentation is consistent with decompression sickness. "
        "Recommend supplemental oxygen and continued monitoring. "
        "Document vitals and prepare a status report for ground."
    ),
    "blast": (
        "Mock response: presentation is consistent with blast overpressure injury. "
        "Assess for pneumothorax using available imaging before further intervention. "
        "Document findings and prepare a status report for ground."
    ),
    "cardiac": (
        "Mock response: presentation is consistent with an unstable arrhythmia. "
        "Recommend restraint, oxygen, and preparing defibrillation equipment. "
        "Document findings and prepare a status report for ground."
    ),
    "default": (
        "Mock response: unable to determine a scenario-specific match; "
        "returning a generic placeholder response for pipeline testing."
    ),
}


def _mock_text_for_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    for key, text in _MOCK_LIBRARY.items():
        if key != "default" and key in lowered:
            return text
    return _MOCK_LIBRARY["default"]


class ModelClient:
    """Unified entry point for querying vertex / edge / mock backends with
    the same call signature, so eval_pipeline.py doesn't need to know which
    backend it's running against."""

    def __init__(
        self,
        backend: Literal["vertex", "edge", "mock"] = "vertex",
        project: Optional[str] = None,
        location: str = "us-central1",
        edge_base_url: Optional[str] = None,
    ):
        self.backend = backend
        self.edge_base_url = (edge_base_url or os.environ.get("EDGE_MODEL_BASE_URL") or "http://localhost:11434").rstrip("/")
        self._gemini_cache: dict = {}

        if backend == "vertex":
            self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not self.project:
                raise ValueError(
                    "Set GOOGLE_CLOUD_PROJECT env var or pass project= for vertex backend."
                )
            self.location = location
            import vertexai  # imported lazily so edge/mock modes don't require the SDK

            vertexai.init(project=self.project, location=self.location)
        elif backend == "edge":
            pass  # no setup required beyond edge_base_url; connection is checked per-call
        elif backend == "mock":
            pass  # no setup required
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'vertex', 'edge', or 'mock'.")

    # ---- Unified entry point -------------------------------------------
    def query(
        self,
        model_id: str,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> ModelResponse:
        if self.backend == "vertex":
            return self._query_vertex(model_id, prompt, system_instruction, temperature, max_output_tokens)
        if self.backend == "edge":
            return self._query_edge(model_id, prompt, system_instruction, temperature, max_output_tokens)
        if self.backend == "mock":
            return self._query_mock(model_id, prompt, system_instruction)
        raise ValueError(f"Unknown backend '{self.backend}'.")

    # ---- vertex: first-party Gemini models ------------------------------
    def _query_vertex(
        self,
        model_id: str,
        prompt: str,
        system_instruction: Optional[str],
        temperature: float,
        max_output_tokens: int,
    ) -> ModelResponse:
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        start = time.time()
        try:
            if model_id not in self._gemini_cache:
                self._gemini_cache[model_id] = GenerativeModel(
                    model_id, system_instruction=system_instruction
                )
            model = self._gemini_cache[model_id]
            resp = model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
            text = resp.text if resp.candidates else ""
            return ModelResponse(
                model_id=model_id, backend="vertex", prompt=prompt, text=text,
                latency_seconds=time.time() - start,
            )
        except Exception as e:  # noqa: BLE001 — surfaced in report, not swallowed
            return ModelResponse(
                model_id=model_id, backend="vertex", prompt=prompt, text="",
                latency_seconds=time.time() - start, error=str(e),
            )

    def query_vertex_endpoint(
        self,
        endpoint_resource_name: str,
        prompt: str,
        model_label: Optional[str] = None,
    ) -> ModelResponse:
        """Query a Vertex AI Model Garden endpoint you've already deployed
        (e.g. a fine-tuned/deployed MedGemma). `endpoint_resource_name`
        looks like: projects/PROJECT/locations/LOCATION/endpoints/ENDPOINT_ID
        Only valid when backend == "vertex"."""
        from google.cloud import aiplatform

        start = time.time()
        label = model_label or endpoint_resource_name
        try:
            endpoint = aiplatform.Endpoint(endpoint_resource_name)
            instances = [{"prompt": prompt, "max_tokens": 1024, "temperature": 0.2}]
            prediction = endpoint.predict(instances=instances)
            text = str(prediction.predictions[0])  # response shape varies by model container
            return ModelResponse(
                model_id=label, backend="vertex", prompt=prompt, text=text,
                latency_seconds=time.time() - start,
            )
        except Exception as e:  # noqa: BLE001
            return ModelResponse(
                model_id=label, backend="vertex", prompt=prompt, text="",
                latency_seconds=time.time() - start, error=str(e),
            )

    # ---- edge: local disconnected inference ------------------------------
    def _query_edge(
        self,
        model_id: str,
        prompt: str,
        system_instruction: Optional[str],
        temperature: float,
        max_output_tokens: int,
    ) -> ModelResponse:
        """Targets an Ollama-compatible HTTP API by default
        (POST {base_url}/api/generate). If you're running a local
        LiteRT/ONNX HTTP container instead, point --edge-base-url at it and
        adjust the request/response shape below to match your container's
        contract — the important part for the benchmark is that this call
        makes no outbound network request."""
        start = time.time()
        full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
        try:
            resp = requests.post(
                f"{self.edge_base_url}/api/generate",
                json={
                    "model": model_id,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_output_tokens,
                    },
                },
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "")
            return ModelResponse(
                model_id=model_id, backend="edge", prompt=prompt, text=text,
                latency_seconds=time.time() - start,
            )
        except Exception as e:  # noqa: BLE001
            return ModelResponse(
                model_id=model_id, backend="edge", prompt=prompt, text="",
                latency_seconds=time.time() - start,
                error=(
                    f"{e} — is a local model server running at {self.edge_base_url}? "
                    "For Ollama: `ollama serve` and `ollama pull <model>` first."
                ),
            )

    # ---- mock: deterministic offline responses ---------------------------
    def _query_mock(
        self, model_id: str, prompt: str, system_instruction: Optional[str]
    ) -> ModelResponse:
        # Hash included only to make it obvious in results that this is
        # synthetic, reproducible output, not a real model call.
        digest = hashlib.sha256((model_id + prompt).encode("utf-8")).hexdigest()[:8]
        text = f"[mock:{digest}] {_mock_text_for_prompt(prompt)}"
        return ModelResponse(
            model_id=model_id, backend="mock", prompt=prompt, text=text,
            latency_seconds=0.0,
        )
