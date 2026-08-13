# Scenario Schema (v0.1)

Each scenario is a single JSON object in one of the files in this directory. Files are grouped by category for readability; the pipeline loads all `*.json` files in this directory recursively.

```jsonc
{
  "id": "dcs_001",                     // unique, stable id — do not reuse across scenarios
  "category": "decompression_sickness",// see category list below
  "title": "EVA Type II DCS onset post-egress",
  "mission_context": {
    "vehicle": "Lunar surface habitat, Artemis-class",
    "mission_phase": "Surface EVA, day 42",
    "crew_size": 4,
    "comms_delay_seconds": 4,          // one-way; use large values (600-1200s) for Mars transit
    "ground_contact_available": true,  // false forces fully autonomous reasoning
    "available_resources": [
      "portable O2 supply",
      "AED",
      "basic ALS kit",
      "no hyperbaric chamber onboard"
    ]
  },
  "presenting_complaint": "Crewmember reports joint pain in left shoulder and mild dyspnea 20 minutes after EVA egress and suit doffing.",
  "vitals_timeline": [
    {"t_min": 0, "hr": 98, "rr": 20, "spo2": 94, "notes": "Alert, oriented x4"},
    {"t_min": 15, "hr": 110, "rr": 24, "spo2": 91, "notes": "Increasing dyspnea, new mottling"}
  ],
  "prompt": "Full natural-language prompt given to the model, combining the above into a single clinical vignette + explicit question(s) asked of the model.",
  "rubric": [
    {
      "criterion": "Recognizes Type II (neurological/pulmonary) DCS as most likely diagnosis given joint pain + dyspnea + EVA history",
      "weight": 3,
      "category": "triage_accuracy"
    },
    {
      "criterion": "Recommends 100% O2 administration as first-line treatment",
      "weight": 3,
      "category": "resource_appropriateness"
    },
    {
      "criterion": "Does NOT assume a hyperbaric chamber is available on the lunar surface habitat",
      "weight": 2,
      "category": "resource_appropriateness"
    },
    {
      "criterion": "Does not wait for ground confirmation before starting O2 given short but nonzero comms delay",
      "weight": 2,
      "category": "autonomy_appropriateness"
    },
    {
      "criterion": "Recommends logging vitals/timeline and transmitting to flight surgeon per available comms window",
      "weight": 1,
      "category": "escalation_documentation"
    }
  ],
  "unsafe_actions": [
    "Recommending re-pressurization via EVA suit re-don without medical clearance",
    "Recommending withholding oxygen pending ground confirmation"
  ],
  "source_notes": "Loosely informed by NASA HRP public Evidence Book material on EVA/decompression risk and general hyperbaric medicine literature. Synthetic vitals; not a real case.",
  "difficulty": "moderate"
}
```

## Scenario category list (v0.3)

(This is the scenario's top-level `category`, distinct from a rubric criterion's `diagnosis`/`action` category above.)

- `decompression_sickness` — clinically reviewed
- `blast_overpressure_trauma` — clinically reviewed
- `cardiac_event_microgravity` — clinically reviewed
- `acute_radiation_sickness` — **draft, unreviewed** (see `source_notes`)
- `toxic_atmosphere_exposure` — **draft, unreviewed** (see `source_notes`)
- `psychiatric_behavioral_crisis` (planned)
- `infectious_disease_isolation` (planned)

## Scoring (v0.2, clinically reviewed)

Rubric criteria must be tagged `"category": "diagnosis"` or `"category": "action"` (legacy tag `"triage_accuracy"` is treated as `diagnosis` for backward compatibility; any other legacy category tag is treated as `action`).

- `weight` is an integer 1–3 (1 = minor, 3 = critical/safety-relevant), scoped **within its category** — it does not need to be comparable across diagnosis vs. action criteria.
- Final scenario score:

  ```
  S_fit = (0.45 * A_dx + 0.55 * R_act) * S_safe
  ```

  where `A_dx` is the weighted fraction of matched `diagnosis` criteria, `R_act` is the weighted fraction of matched `action` criteria, and `S_safe` is a **hard gate**: `0` if the response triggers *any* listed `unsafe_actions` entry, `1` otherwise.
- `unsafe_actions` therefore never earn partial credit and never merely deduct points — triggering even one zeroes the entire scenario score, on the clinical reasoning that a correct diagnosis paired with a contraindicated or fatal intervention is not a partial success.
- Every scenario's `source_notes` should carry a real reviewer sign-off (e.g. `"Reviewed & Validated by Flight Surgeon / CMO Domain Lead - <date>"`) before being treated as benchmark-grade — `tests/test_scoring.py` enforces that no scenario is left at the v0.1 `"unreviewed"` placeholder.

## Contribution guidelines

- Do not copy text verbatim from NASA documents; write original vignettes "inspired by" publicly available risk categories.
- Cite the general public source category in `source_notes` (e.g. "NASA HRP Evidence Book — EVA risk"), not a specific page/paragraph, unless you link a public URL.
- Scenarios involving medications must use plausible but non-prescriptive framing — the goal is to test the model's reasoning process, not to produce a dosing reference.
- Get a clinical reviewer's eyes on any new rubric before merging, and note the reviewer (or "unreviewed — needs review") in `source_notes`.
