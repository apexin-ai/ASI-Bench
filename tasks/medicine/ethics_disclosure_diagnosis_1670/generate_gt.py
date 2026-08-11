"""Generate a seed-dependent, multi-case disclosure-ethics audit."""

from __future__ import annotations

import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path


TASK_ID = "medicine.ethics_disclosure_diagnosis_1670"
DEFAULT_PARAMS = {"seed": 0}
CAPACITY_STATES = ("capable", "incapable", "uncertain_reassess")
DISCLOSURE_PLANS = (
    "disclose_to_patient_now",
    "honor_patient_delegation",
    "defer_details_until_capacity_restored",
    "supported_disclosure_with_accommodations",
)
SURROGATE_ROLES = (
    "support_only",
    "no_current_decisional_authority",
    "authority_for_treatment_only",
    "authority_for_all_current_decisions",
)
ACTION_IDS = (
    "interview_patient_privately",
    "use_professional_interpreter",
    "assess_information_preference_capacity",
    "assess_treatment_capacity",
    "elicit_patient_information_preferences",
    "explore_family_concerns_without_promising_secrecy",
    "provide_paced_supported_disclosure",
    "honor_explicit_patient_delegation",
    "treat_reversible_delirium",
    "reassess_capacity_after_stabilization",
    "use_surrogate_for_urgent_treatment",
    "ethics_consult_if_conflict_persists",
    "document_capacity_preferences_and_plan",
)
PRINCIPLE_IDS = (
    "autonomy",
    "right_not_to_know",
    "decision_specific_capacity",
    "supported_decision_making",
    "substituted_judgment",
    "best_interests",
    "confidentiality",
    "cultural_humility_without_stereotyping",
    "least_restrictive_alternative",
)


CASES = [
    {
        "key": "capable_direct",
        "facts": [
            ("family", "The daughter asks the team to tell only the family because the patient will 'lose hope.'", "key"),
            ("capacity", "In a private interview, the patient accurately explains the biopsy result may be cancer, compares treatment with comfort care, and explains how either choice fits her goals.", "key"),
            ("preference", "She says, 'Please tell me the result first. I will decide what my children hear.'", "key"),
            ("proxy", "Her son is named healthcare proxy; the document activates only upon loss of capacity.", "support"),
            ("risk", "The family predicts sadness but reports no suicidality, psychosis, or prior dangerous reaction to medical news.", "support"),
            ("noise", "A copied nursing note calls the son the 'current decision maker' without a capacity assessment.", "noise"),
        ],
        "info_capacity": "capable",
        "treatment_capacity": "capable",
        "plan": "disclose_to_patient_now",
        "surrogate": "no_current_decisional_authority",
        "actions": ["interview_patient_privately", "explore_family_concerns_without_promising_secrecy", "provide_paced_supported_disclosure", "document_capacity_preferences_and_plan"],
        "principles": ["autonomy", "confidentiality", "cultural_humility_without_stereotyping"],
        "evidence": ["capacity", "preference", "risk"],
        "contingency": {"trigger": "patient_loses_treatment_capacity", "action": "activate_proxy_for_treatment_using_substituted_judgment"},
    },
    {
        "key": "capable_delegation",
        "facts": [
            ("family", "The spouse asks clinicians to route detailed cancer information through him.", "support"),
            ("preference", "Six months before illness, and again privately today, the patient states: 'Tell me only whether treatment is needed; give details and choices to my spouse, who will explain what I ask for.'", "key"),
            ("capacity", "The patient understands that delegating information may leave her without prognostic details, appreciates that she can revoke the choice, and consistently communicates the same preference.", "key"),
            ("treatment", "When offered two treatment summaries in her preferred limited-detail format, she understands benefits and burdens and makes a stable choice.", "key"),
            ("distress", "She becomes anxious when clinicians repeatedly press her to hear staging details but remains coherent and non-delusional.", "support"),
            ("noise", "A resident writes that all nondisclosure is incompatible with autonomy.", "noise"),
        ],
        "info_capacity": "capable",
        "treatment_capacity": "capable",
        "plan": "honor_patient_delegation",
        "surrogate": "support_only",
        "actions": ["interview_patient_privately", "elicit_patient_information_preferences", "honor_explicit_patient_delegation", "document_capacity_preferences_and_plan"],
        "principles": ["autonomy", "right_not_to_know", "supported_decision_making"],
        "evidence": ["preference", "capacity", "treatment"],
        "contingency": {"trigger": "patient_requests_details", "action": "disclose_requested_details_directly_to_patient"},
    },
    {
        "key": "delirium_urgent",
        "facts": [
            ("baseline", "At a clinic visit two weeks ago, the patient requested direct disclosure and demonstrated intact understanding.", "support"),
            ("delirium", "Today, with cholangitis and fever, she alternates between believing she is at home and in hospital and cannot retain the diagnosis for five minutes.", "key"),
            ("treatment", "Urgent biliary drainage is recommended tonight; waiting several days creates substantial sepsis risk.", "key"),
            ("disclosure", "Detailed tumor-genomics counseling is clinically nonurgent and can wait until infection and delirium improve.", "key"),
            ("proxy", "The properly designated proxy describes the patient's longstanding goal of accepting short-term burdens to regain independence.", "support"),
            ("noise", "The proxy asks that the cancer diagnosis never be discussed even after recovery.", "noise"),
        ],
        "info_capacity": "incapable",
        "treatment_capacity": "incapable",
        "plan": "defer_details_until_capacity_restored",
        "surrogate": "authority_for_all_current_decisions",
        "actions": ["treat_reversible_delirium", "use_surrogate_for_urgent_treatment", "reassess_capacity_after_stabilization", "document_capacity_preferences_and_plan"],
        "principles": ["decision_specific_capacity", "substituted_judgment", "least_restrictive_alternative"],
        "evidence": ["delirium", "treatment", "disclosure"],
        "contingency": {"trigger": "capacity_returns", "action": "surrogate_authority_ends_and_direct_disclosure_resumes"},
    },
    {
        "key": "split_capacity",
        "facts": [
            ("cognition", "The patient has mild cognitive impairment and needs repetition, but consistently identifies whom she trusts and whether she wants diagnostic information.", "key"),
            ("preference", "Privately and without prompting, she asks for the diagnosis in plain language with her niece present.", "key"),
            ("treatment", "She cannot compare the delayed toxicities of three chemotherapy regimens or explain how they affect her stated priority of living independently.", "key"),
            ("support", "A visual decision aid and teach-back improve recall of the diagnosis but not comparison of the treatment alternatives.", "support"),
            ("proxy", "Her niece is the designated proxy and accurately states the patient's prior refusal of highly toxic treatment.", "support"),
            ("noise", "A binary capacity checkbox in the admission header is marked 'incapable for all decisions.'", "noise"),
        ],
        "info_capacity": "capable",
        "treatment_capacity": "incapable",
        "plan": "supported_disclosure_with_accommodations",
        "surrogate": "authority_for_treatment_only",
        "actions": ["assess_information_preference_capacity", "assess_treatment_capacity", "provide_paced_supported_disclosure", "document_capacity_preferences_and_plan"],
        "principles": ["decision_specific_capacity", "supported_decision_making", "substituted_judgment"],
        "evidence": ["cognition", "preference", "treatment"],
        "contingency": {"trigger": "treatment_understanding_improves_with_support", "action": "return_treatment_decision_to_patient"},
    },
    {
        "key": "language_coercion",
        "facts": [
            ("language", "The patient speaks limited English. Her adult son has answered every question and refuses a certified interpreter.", "key"),
            ("family", "The son says family custom requires that he receive the diagnosis and decide what his mother is told.", "support"),
            ("behavior", "When the son leaves briefly, the patient quietly asks in her language whether the doctors can speak with her alone.", "key"),
            ("record", "No prior information preference, capacity assessment, or valid delegation is documented.", "key"),
            ("clinical", "The result is important but no treatment decision is required in the next several hours.", "support"),
            ("noise", "The admission template labels the son 'preferred interpreter and surrogate.'", "noise"),
        ],
        "info_capacity": "uncertain_reassess",
        "treatment_capacity": "uncertain_reassess",
        "plan": "supported_disclosure_with_accommodations",
        "surrogate": "no_current_decisional_authority",
        "actions": ["interview_patient_privately", "use_professional_interpreter", "assess_information_preference_capacity", "elicit_patient_information_preferences"],
        "principles": ["autonomy", "supported_decision_making", "cultural_humility_without_stereotyping"],
        "evidence": ["language", "behavior", "record"],
        "contingency": {"trigger": "patient_capably_delegates_information", "action": "honor_delegation_while_preserving_revocation_right"},
    },
]


OUTPUT_CONTRACT = """Write `ethics_plan.json` at the workspace root as valid JSON:

```json
{"cases": [{
  "case_id": "C-...",
  "capacity_for_information_choice": "capable|incapable|uncertain_reassess",
  "capacity_for_treatment": "capable|incapable|uncertain_reassess",
  "disclosure_plan": "one allowed disclosure-plan ID",
  "surrogate_role": "one allowed surrogate-role ID",
  "immediate_actions": ["at most four action IDs"],
  "ethical_basis": ["exactly three principle IDs"],
  "decisive_evidence": ["exactly three event IDs"],
  "contingency": {"trigger": "one trigger ID", "action": "one contingency-action ID"},
  "justification": "maximum 500 characters"
}]}
```

Assess every case exactly once. Use only IDs printed in the dossier. Capacity is
decision-specific; do not infer global incapacity from diagnosis, age, distress,
or a checkbox. A valid patient choice to delegate information is different from
a family-imposed request. Prioritize clinically necessary actions rather than
listing every generally reasonable step."""


PROMPTS = {
    "b1": """# B1 — guided disclosure-ethics audit

Review `data/ethics_dossier.md`. Apply the four-abilities model separately to
the information-preference decision and the treatment decision. Distinguish a
capable patient's delegation/right not to know from family-imposed secrecy;
activate proxy authority only for decisions the patient cannot make. Use
supported decision-making and the least restrictive response.

{contract}
""",
    "b2": """# B2 — longitudinal disclosure-ethics audit

Review `data/ethics_dossier.md`. Resolve each patient's information-disclosure,
capacity, and surrogate-authority questions and give the immediate plan.

{contract}
""",
    "b3": """# B3 — clinical ethics record reconciliation

Audit the four shuffled records in `data/ethics_dossier.md` and determine the
ethically and clinically appropriate decisions for each case.

{contract}
""",
    "b4": """# B4 — adversarial clinical ethics audit

Audit `data/ethics_dossier.md`. Some chart labels collapse distinct decisions,
some family requests conflict with the patient's authority, and some apparently
similar nondisclosure requests are ethically valid. Reconcile the evidence and
state the action that follows if circumstances change.

{contract}
""",
}


def _case_id(seed: int, index: int) -> str:
    return f"C-{(seed * 6151 + index * 1877 + 23003) % 90000 + 10000:05d}"


def _reference(case: dict) -> dict:
    return {
        "case_id": case["case_id"],
        "valid_event_ids": list(case["event_ids"].values()),
        "capacity_for_information_choice": case["info_capacity"],
        "capacity_for_treatment": case["treatment_capacity"],
        "disclosure_plan": case["plan"],
        "surrogate_role": case["surrogate"],
        "immediate_actions": case["actions"],
        "ethical_basis": case["principles"],
        "decisive_evidence": [case["event_ids"][key] for key in case["evidence"]],
        "contingency": case["contingency"],
        "justification": "Reference decisions are encoded in ground_truth.json.",
    }


def _render_dossier(cases: list[dict], rng: random.Random) -> str:
    lines = ["# De-identified disclosure-ethics review queue", "", "Allowed disclosure-plan IDs:"]
    lines.extend(f"- `{x}`" for x in DISCLOSURE_PLANS)
    lines.extend(["", "Allowed surrogate-role IDs:"])
    lines.extend(f"- `{x}`" for x in SURROGATE_ROLES)
    lines.extend(["", "Allowed immediate-action IDs:"])
    lines.extend(f"- `{x}`" for x in ACTION_IDS)
    lines.extend(["", "Allowed ethical-principle IDs:"])
    lines.extend(f"- `{x}`" for x in PRINCIPLE_IDS)
    contingency_triggers = [case["contingency"]["trigger"] for case in cases]
    contingency_actions = [case["contingency"]["action"] for case in cases]
    rng.shuffle(contingency_triggers)
    rng.shuffle(contingency_actions)
    lines.extend(["", "Allowed contingency-trigger IDs for this instance:"])
    lines.extend(f"- `{x}`" for x in contingency_triggers)
    lines.extend(["", "Allowed contingency-action IDs for this instance:"])
    lines.extend(f"- `{x}`" for x in contingency_actions)
    lines.extend(["", "Records are shuffled. Event IDs are local to one case."])
    for case in cases:
        lines.extend(["", f"## Case {case['case_id']}", ""])
        facts = list(case["facts"])
        rng.shuffle(facts)
        for key, text, _role in facts:
            lines.append(f"- `{case['event_ids'][key]}` — {text}")
    return "\n".join(lines) + "\n"


def generate(output_dir: Path, params: dict) -> dict:
    p = {**DEFAULT_PARAMS, **params}
    seed = int(p["seed"])
    rng = random.Random(seed)
    started = time.time()
    data_dir = output_dir / "data"
    ref_dir = output_dir / "reference"
    data_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    chosen = [deepcopy(case) for case in rng.sample(CASES, 4)]
    for index, case in enumerate(chosen):
        case["case_id"] = _case_id(seed, index)
        keys = [fact[0] for fact in case["facts"]]
        case["event_ids"] = {key: f"E{token}" for key, token in zip(keys, rng.sample(range(101, 999), len(keys)))}
    rng.shuffle(chosen)
    (data_dir / "ethics_dossier.md").write_text(_render_dossier(chosen, rng), encoding="utf-8")
    for level, template in PROMPTS.items():
        (output_dir / f"prompt_{level}.md").write_text(template.format(contract=OUTPUT_CONTRACT).strip() + "\n", encoding="utf-8")

    references = [_reference(case) for case in chosen]
    ground_truth = {
        "schema_version": 1,
        "capacity_states": list(CAPACITY_STATES),
        "disclosure_plans": list(DISCLOSURE_PLANS),
        "surrogate_roles": list(SURROGATE_ROLES),
        "action_ids": list(ACTION_IDS),
        "principle_ids": list(PRINCIPLE_IDS),
        "cases": references,
    }
    (ref_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    clean = [{key: value for key, value in row.items() if key != "valid_event_ids"} for row in references]
    (ref_dir / "ethics_plan.json").write_text(json.dumps({"cases": clean}, indent=2), encoding="utf-8")
    meta = {"task_id": TASK_ID, "parameters": p, "case_count": 4, "elapsed_seconds": time.time() - started}
    (output_dir / "instance_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def self_check(instance_dir: Path, *_args) -> dict:
    errors: list[str] = []
    try:
        gt = json.loads((instance_dir / "reference" / "ground_truth.json").read_text(encoding="utf-8"))
        assert len(gt["cases"]) == 4
        assert all(len(row["decisive_evidence"]) == 3 for row in gt["cases"])
        assert all(len(row["immediate_actions"]) <= 4 for row in gt["cases"])
    except Exception as exc:
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
