# Contributing

Thanks for considering a contribution. This benchmark is only as good as its scenarios and its scrutiny.

## Adding scenarios

1. Follow `data/scenarios/schema.md` exactly.
2. Write original vignettes — do not copy text from NASA documents or any copyrighted source. "Inspired by" a public risk category is fine; verbatim reproduction is not.
3. Fill `source_notes` honestly, including reviewer status. If unreviewed by a clinician, say so.
4. Keep `unsafe_actions` realistic and specific — vague entries make scoring meaningless.
5. Open an issue describing the scenario/category before submitting a large PR so reviewers can weigh in early.

## Adding model backends

- `src/vertex_client.py` currently supports first-party Gemini models and Model Garden endpoint-deployed open models. PRs adding support for other Vertex-compatible backends are welcome — keep the `ModelResponse` interface consistent.

## Reporting issues with scoring

- If you believe `score_keyword` or `score_llm_judge` mis-scored a response, open an issue with the scenario id, model id, and the raw response text (from `results/responses_*.json`) so it's reproducible.

## Code of conduct

Be respectful. This repo touches medical and safety topics — assume good faith, cite sources, and flag uncertainty rather than overclaiming.
