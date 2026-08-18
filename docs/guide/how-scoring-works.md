# How Scoring Works

ASI-Bench has two explicit scoring contracts: seed31415 publishes references
for local scoring, while seed42 keeps references private and uses authenticated
website scoring.

## The split

| Layer | Public? | What it is |
|---|---|---|
| Framework — runner, sandboxes, output collection and submission | **Public** | The machinery for executing agents and packaging their outputs. |
| Task **metadata + prompts + input data** | **Public** | What an agent needs to attempt a task. |
| Task **scoring/output contract + custom scorers** | **Public** | Auditable gates, weights, tolerances, and scorer implementation, without generation or reference content. |
| seed31415 reference answers | **Public on Hugging Face** | Reproducible local scoring with GitHub scorers. |
| seed42 reference answers, all `generate_gt.py`, generation settings, reference specs, private solver assets | **Private** | Website-only answer material and everything needed to create it. |

The ASI-Bench website owns seed42 evaluation and uses private references.
seed31415 local scoring is deliberately public but marked non-official.

## Who scores, and when

1. You run either seed in produce-only mode (`asibench run`).
2. For seed31415, `asibench score --repo seed31415` uses the pulled public
   references and this checkout's GitHub scorers, writing a separate report.
3. For seed42, `asibench login` identifies the submitter and `asibench submit` uploads an
   authenticated draft to the ASI-Bench website.
4. You **confirm** the submission in the browser; it enters the website's scoring
   queue and is evaluated against private task material.
5. The confirmed, officially scored run can be published to the leaderboard.

**Self-reported scores are never trusted.** A run only appears on the leaderboard
after scoring through the ASI-Bench website.

## Why the seeds differ

seed31415 is the open evaluation split: public references make scorer behavior
fully reproducible. seed42 is the protected evaluation split: public scoring
logic remains auditable, but references are only available to the website.
Neither split publishes GT generators or private solver assets.

## Reproducibility

Ground-truth answers are deterministic: given the same parameters and random seed,
a task's `generate_gt.py` produces the same reference every time. Scoring compares
your outputs to that reference **with tolerances**, so minor, environment-level
floating-point differences do not change the score. Runs also record full
provenance (agent, model, effort, sandbox, framework version) so a result can be
reproduced and fairly compared to others in the same bucket.
