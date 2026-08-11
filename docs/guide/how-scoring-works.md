# How Scoring Works

ASI-Bench separates public benchmark execution from official website scoring.
This repository does not calculate benchmark scores; authenticated runs are
submitted to `https://asibench.apexin.ai/` for scoring.

## The split

| Layer | Public? | What it is |
|---|---|---|
| Framework — runner, sandboxes, output collection and submission | **Public** | The machinery for executing agents and packaging their outputs. |
| Task **metadata + prompts + input data** | **Public** | What an agent needs to attempt a task. |
| Task **reference answers, `generate_gt.py`, `task_eval.yaml`, custom scorers** | **Private** | The answer key and the exact scoring rules for each graded task. |

The ASI-Bench website owns the official evaluation workflow and uses private
reference material that is not distributed with this repository.

## Who scores, and when

1. You run in produce-only mode (`asibench run`).
2. `asibench login` identifies the submitter, and `asibench submit` uploads an
   authenticated draft to the ASI-Bench website.
3. You **confirm** the submission in the browser; it enters the website's scoring
   queue and is evaluated against private task material.
4. The confirmed, officially scored run can be published to the leaderboard.

**Self-reported scores are never trusted.** A run only appears on the leaderboard
after scoring through the ASI-Bench website.

## Why produce-only for external runners

External runners do not have the reference answers or scoring config for graded
tasks, so they cannot compute an official score locally — that is intentional. You
produce outputs and submit them to the website. This prevents gaming the benchmark
by reading or reverse-engineering the answers.

## Reproducibility

Ground-truth answers are deterministic: given the same parameters and random seed,
a task's `generate_gt.py` produces the same reference every time. Scoring compares
your outputs to that reference **with tolerances**, so minor, environment-level
floating-point differences do not change the score. Runs also record full
provenance (agent, model, effort, sandbox, framework version) so a result can be
reproduced and fairly compared to others in the same bucket.
