# B4 — adversarial chart reconciliation under a budget

Audit `data/clinical_dossier.md`. Each chart contains a plausible but
potentially unreliable diagnostic label. Reconstruct the chronology, separate
observations from copied assertions, quantify uncertainty, and select actions
within the six-unit budget per patient. Identify the single recorded event whose
removal most changes each leading diagnosis.

Create `final_assessment.json` using the schema described in the generated
prompt. Assess every patient exactly once.
