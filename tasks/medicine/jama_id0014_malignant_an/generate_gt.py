"""Generate a multi-patient dermatology-oncology reasoning instance.

The task deliberately varies the true diagnosis.  Agents must reconcile noisy
longitudinal evidence and make budgeted next-step decisions; there is no fixed
answer letter or keyword-completion rubric.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path


TASK_ID = "medicine.jama_id0014_malignant_an"
DEFAULT_PARAMS = {"seed": 0}
DIAGNOSES = (
    "malignant_acanthosis_nigricans",
    "benign_acanthosis_nigricans",
    "drug_induced_acanthosis_nigricans",
    "confluent_reticulated_papillomatosis",
    "cutaneous_t_cell_lymphoma",
)
ACTION_COSTS = {
    "egd_with_biopsy": 3,
    "ct_chest_abdomen_pelvis": 3,
    "repeat_skin_biopsy_with_tcr": 3,
    "metabolic_risk_assessment": 1,
    "medication_reconciliation": 1,
    "stop_or_substitute_culprit_drug": 1,
    "trial_oral_minocycline": 2,
    "fungal_scraping_koh": 1,
    "routine_dermatology_followup": 1,
}


ARCHETYPES = [
    {
        "key": "gastric_man",
        "diagnosis": "malignant_acanthosis_nigricans",
        "occult_primary": "gastric_adenocarcinoma",
        "disposition": "urgent_malignancy_workup",
        "probabilities": [0.78, 0.07, 0.03, 0.04, 0.08],
        "actions": ["egd_with_biopsy", "ct_chest_abdomen_pelvis"],
        "events": [
            ("onset", "Over 11 weeks, new velvety plaques spread from both axillae to groin, palms, lips, gingiva, and conjunctiva.", "for"),
            ("systemic", "Unintentional 7.8 kg loss, early satiety, and persistent epigastric discomfort developed during the same interval.", "for"),
            ("markers", "CEA 31 ng/mL, CA 19-9 218 U/mL, and CA 72-4 49 U/mL are elevated on repeat sampling.", "for"),
            ("histology", "Two sites show papillomatosis, hyperkeratosis, and basal hyperpigmentation without epidermotropic atypical lymphocytes.", "for"),
            ("metabolic", "BMI is 29; HbA1c is 5.5%; there is no family history of similar plaques.", "against"),
            ("noise", "An urgent-care note labels the eruption 'frictional intertrigo' without documenting a full skin examination.", "neutral"),
        ],
        "counterfactual": ("systemic", "benign_acanthosis_nigricans"),
    },
    {
        "key": "insulin_ban",
        "diagnosis": "benign_acanthosis_nigricans",
        "occult_primary": "none",
        "disposition": "metabolic_management",
        "probabilities": [0.04, 0.80, 0.08, 0.05, 0.03],
        "actions": ["metabolic_risk_assessment", "routine_dermatology_followup"],
        "events": [
            ("course", "Axillary and groin plaques have darkened gradually over 8 years without palmar or mucosal disease.", "for"),
            ("family", "The patient's mother and two siblings developed nearly identical flexural plaques in adulthood.", "for"),
            ("metabolic", "BMI 37, HbA1c 7.1%, fasting insulin 32 microU/mL, and triglycerides 286 mg/dL indicate insulin resistance.", "for"),
            ("review", "There is no weight loss, early satiety, lymphadenopathy, fever, or night sweats.", "against"),
            ("prior", "A CT performed for renal colic 4 months ago showed no mass; CBC and liver tests are normal.", "against"),
            ("noise", "A copied problem-list entry says 'rule out paraneoplastic syndrome' but predates the family history and laboratory review.", "neutral"),
        ],
        "counterfactual": ("course", "malignant_acanthosis_nigricans"),
    },
    {
        "key": "niacin_an",
        "diagnosis": "drug_induced_acanthosis_nigricans",
        "occult_primary": "none",
        "disposition": "medication_change_and_followup",
        "probabilities": [0.08, 0.20, 0.62, 0.06, 0.04],
        "actions": ["medication_reconciliation", "stop_or_substitute_culprit_drug", "metabolic_risk_assessment"],
        "events": [
            ("drug", "Extended-release niacin was increased from 500 to 2000 mg/day 7 months before the eruption began.", "for"),
            ("timeline", "Flexural velvety plaques appeared 5 months ago and worsened as fasting glucose rose from 96 to 132 mg/dL.", "for"),
            ("distribution", "There are no oral, conjunctival, palmar, or nail findings.", "against"),
            ("systemic", "Weight is stable; review for gastrointestinal and constitutional symptoms is negative.", "against"),
            ("histology", "Biopsy shows hyperkeratosis and papillomatosis without atypical lymphocytes or interface change.", "for"),
            ("noise", "The referral diagnosis is 'new malignant acanthosis nigricans'; the referring clinician did not list nonprescription drugs.", "neutral"),
        ],
        "counterfactual": ("drug", "benign_acanthosis_nigricans"),
    },
    {
        "key": "carp",
        "diagnosis": "confluent_reticulated_papillomatosis",
        "occult_primary": "none",
        "disposition": "dermatologic_treatment",
        "probabilities": [0.03, 0.11, 0.03, 0.78, 0.05],
        "actions": ["fungal_scraping_koh", "trial_oral_minocycline"],
        "events": [
            ("morphology", "Brown scaly papules coalesce centrally and form a reticulated edge over the intermammary chest and upper back.", "for"),
            ("distribution", "The axillary vaults are largely spared; there is no mucosal, palmar, or nail involvement.", "against"),
            ("course", "The eruption evolved over 18 months in a healthy 24-year-old with stable weight and normal HbA1c.", "for"),
            ("fungal", "A prior empiric azole course produced no change; no KOH preparation was documented.", "for"),
            ("histology", "Biopsy shows compact hyperkeratosis, papillomatosis, focal acanthosis, and increased basal melanin.", "for"),
            ("noise", "Automated coding maps 'papillomatosis' to acanthosis nigricans in the billing record.", "neutral"),
        ],
        "counterfactual": ("morphology", "benign_acanthosis_nigricans"),
    },
    {
        "key": "ctcl",
        "diagnosis": "cutaneous_t_cell_lymphoma",
        "occult_primary": "none",
        "disposition": "urgent_dermatology_oncology_referral",
        "probabilities": [0.04, 0.08, 0.02, 0.10, 0.76],
        "actions": ["repeat_skin_biopsy_with_tcr"],
        "events": [
            ("morphology", "Pruritic erythematous-brown patches and thin plaques have irregular borders and variable scale on bathing-trunk skin.", "for"),
            ("course", "Lesions have persisted and slowly expanded for 4 years despite topical antifungal and keratolytic therapy.", "for"),
            ("histology", "The newest biopsy shows epidermotropism of atypical CD4-predominant lymphocytes with partial CD7 loss; no spongiosis.", "for"),
            ("clonality", "Two earlier superficial biopsies were nondiagnostic; T-cell receptor clonality has not been performed.", "for"),
            ("distribution", "There is no velvety flexural thickening, insulin resistance, mucosal disease, or tripe-palms morphology.", "against"),
            ("noise", "A primary-care photograph was captioned 'possible acanthosis' before palpation or biopsy.", "neutral"),
        ],
        "counterfactual": ("histology", "confluent_reticulated_papillomatosis"),
    },
]


OUTPUT_CONTRACT = """Create `final_assessment.json` at the workspace root. It must be valid JSON:

```json
{
  "assessments": [
    {
      "case_id": "P-...",
      "primary_diagnosis": "one diagnosis ID",
      "differential_probabilities": {"all five diagnosis IDs": 0.0},
      "evidence_for": ["event IDs, at most 4"],
      "evidence_against": ["event IDs, at most 3"],
      "occult_primary": "gastric_adenocarcinoma or none",
      "next_actions": ["action IDs, total cost <= 6"],
      "disposition": "one disposition ID",
      "counterfactual": {"remove_event_id": "event ID", "revised_diagnosis": "diagnosis ID"},
      "rationale": "maximum 500 characters"
    }
  ]
}
```

Assess every case exactly once. Probabilities must include all five diagnoses,
be finite numbers in [0,1], and sum to 1. Use only IDs defined in the dossier.
Do not order tests merely to spend the budget: unnecessary invasive oncologic
testing can reduce the score."""


PROMPT_TEMPLATES = {
    "b1": """# B1 — guided multi-patient clinical synthesis

Review `data/clinical_dossier.md`. For each patient, distinguish morphology,
tempo, systemic red flags, metabolic or medication context, and histopathology.
Treat copied labels as lower-quality evidence than dated observations. Malignant
AN particularly warrants a search for gastric adenocarcinoma; CTCL requires
clinicopathologic correlation; CARP is primarily a morphology/distribution
diagnosis. Select only actions that change management.

{contract}
""",
    "b2": """# B2 — multi-patient clinical synthesis

Review `data/clinical_dossier.md`. Reconcile the longitudinal record for each
patient, assign a calibrated differential, and choose a cost-constrained plan.
Copied diagnoses may be stale or unsupported.

{contract}
""",
    "b3": """# B3 — chart reconciliation

The raw, shuffled records for four patients are in
`data/clinical_dossier.md`. Determine the diagnoses and management decisions.

{contract}
""",
    "b4": """# B4 — adversarial chart reconciliation under a budget

Audit `data/clinical_dossier.md`. Each chart contains a plausible but
potentially unreliable diagnostic label. Reconstruct the chronology, separate
observations from copied assertions, quantify uncertainty, and select actions
within the six-unit budget per patient. Your counterfactual must identify the
single recorded event whose removal most changes the leading diagnosis.

{contract}
""",
}


def _case_id(seed: int, index: int) -> str:
    return f"P-{(seed * 7919 + index * 1049 + 10007) % 90000 + 10000:05d}"


def _render_dossier(cases: list[dict], rng: random.Random) -> str:
    lines = [
        "# De-identified dermatology case conference dossier",
        "",
        "Action costs (maximum 6 units per patient):",
        *[f"- `{name}`: {cost}" for name, cost in ACTION_COSTS.items()],
        "",
        "Diagnosis IDs:",
        *[f"- `{name}`" for name in DIAGNOSES],
        "",
        "Event IDs are local to a patient. Records are intentionally not in chronological order.",
    ]
    for case in cases:
        lines.extend(["", f"## Patient {case['case_id']}", ""])
        events = list(case["events"])
        rng.shuffle(events)
        for event_id, text, _role in events:
            lines.append(f"- `{case['event_ids'][event_id]}` — {text}")
    return "\n".join(lines) + "\n"


def _reference_assessment(case: dict) -> dict:
    probs = {name: value for name, value in zip(DIAGNOSES, case["probabilities"])}
    for_ids = [case["event_ids"][key] for key, _text, role in case["events"] if role == "for"][:4]
    against_ids = [case["event_ids"][key] for key, _text, role in case["events"] if role == "against"][:3]
    remove_key, revised = case["counterfactual"]
    return {
        "case_id": case["case_id"],
        "valid_event_ids": [case["event_ids"][key] for key, _text, _role in case["events"]],
        "primary_diagnosis": case["diagnosis"],
        "differential_probabilities": probs,
        "evidence_for": for_ids,
        "evidence_against": against_ids,
        "occult_primary": case["occult_primary"],
        "next_actions": case["actions"],
        "disposition": case["disposition"],
        "counterfactual": {
            "remove_event_id": case["event_ids"][remove_key],
            "revised_diagnosis": revised,
        },
        "rationale": "Reference decisions are encoded in ground_truth.json.",
    }


def generate(output_dir: Path, params: dict) -> dict:
    p = {**DEFAULT_PARAMS, **params}
    seed = int(p["seed"])
    rng = random.Random(seed)
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    ref_dir = output_dir / "reference"
    data_dir.mkdir(exist_ok=True)
    ref_dir.mkdir(exist_ok=True)

    chosen = rng.sample(ARCHETYPES, 4)
    cases: list[dict] = []
    for index, source in enumerate(chosen):
        case = deepcopy(source)
        case["case_id"] = _case_id(seed, index)
        event_keys = [event[0] for event in case["events"]]
        tokens = rng.sample(range(101, 999), len(event_keys))
        case["event_ids"] = {key: f"E{token}" for key, token in zip(event_keys, tokens)}
        cases.append(case)
    rng.shuffle(cases)

    (data_dir / "clinical_dossier.md").write_text(_render_dossier(cases, rng), encoding="utf-8")
    for level, template in PROMPT_TEMPLATES.items():
        (output_dir / f"prompt_{level}.md").write_text(
            template.format(contract=OUTPUT_CONTRACT).strip() + "\n", encoding="utf-8"
        )

    reference = {
        "schema_version": 1,
        "diagnoses": list(DIAGNOSES),
        "action_costs": ACTION_COSTS,
        "budget": 6,
        "assessments": [_reference_assessment(case) for case in cases],
    }
    (ref_dir / "ground_truth.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    (ref_dir / "final_assessment.json").write_text(
        json.dumps(
            {
                "assessments": [
                    {key: value for key, value in row.items() if key != "valid_event_ids"}
                    for row in reference["assessments"]
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    meta = {
        "task_id": TASK_ID,
        "parameters": p,
        "case_count": len(cases),
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "instance_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def self_check(instance_dir: Path, *_args) -> dict:
    errors: list[str] = []
    try:
        gt = json.loads((instance_dir / "reference" / "ground_truth.json").read_text(encoding="utf-8"))
        assert len(gt["assessments"]) == 4
        assert all(abs(sum(a["differential_probabilities"].values()) - 1.0) < 1e-9 for a in gt["assessments"])
        assert all(sum(gt["action_costs"][x] for x in a["next_actions"]) <= gt["budget"] for a in gt["assessments"])
    except Exception as exc:  # pragma: no cover - surfaced by pre-submit
        errors.append(str(exc))
    return {"status": "ok" if not errors else "error", "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--params", default="{}")
    args = parser.parse_args()
    result = generate(args.output_dir, json.loads(args.params))
    result["self_check"] = self_check(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
