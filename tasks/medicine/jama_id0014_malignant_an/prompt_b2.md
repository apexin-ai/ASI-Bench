# B2 — multi-patient clinical synthesis

Review `data/clinical_dossier.md`. Reconcile the longitudinal record for each
patient, assign a calibrated differential, and choose a cost-constrained plan.
Copied diagnoses may be stale or unsupported.

Create `final_assessment.json` using the schema described in the generated
prompt. Assess all four cases, cite event IDs, and remain within the six-unit
action budget for each patient.
