# B1 — guided multi-patient clinical synthesis

Review `data/clinical_dossier.md`. For each patient, distinguish morphology,
tempo, systemic red flags, metabolic or medication context, and histopathology.
Treat copied labels as lower-quality evidence than dated observations. Malignant
AN particularly warrants a search for gastric adenocarcinoma; CTCL requires
clinicopathologic correlation; CARP is primarily a morphology/distribution
diagnosis. Select only actions that change management.

Create `final_assessment.json` using the schema described in the generated
prompt. Assess all four cases, provide probabilities for all five diagnoses,
cite event IDs, remain within the six-unit action budget, and supply the most
diagnostically important counterfactual for each patient.
