# Where Things Live

ASI-Bench is spread across a few places. As an external user you interact with
the code on GitHub and fixed-seed datasets on Hugging Face. The answer keys are
kept separately and privately.

## The public pieces

| Where | What it holds | You use it to… |
|---|---|---|
| **GitHub — the `asibench` repo** | The produce-only runner and public task metadata; official prompts are not tracked here | `pip install asibench`, run the benchmark, and package outputs |
| **Hugging Face — seed42** | One official fixed-seed task set with prompts and input data | `asibench task pull --repo seed42` |
| **Hugging Face — seed31415** | The second official fixed-seed task set with prompts and input data | `asibench task pull --repo seed31415` |

```bash
# official fixed-seed tasks (pull and run separately)
asibench task pull --repo seed42 --output-dir hf_instances_seed42/
asibench task pull --repo seed31415 --output-dir hf_instances_seed31415/
```

The `pull` command resolves the dataset location for you; you never need to type
the raw repository path. There is intentionally no default: `seed42` and
`seed31415` are the only official contracts, and every pull must select one.

## Task catalog versus instance manifest

| File | Stored at | Purpose | Required for a produce-only run? |
|---|---|---|---|
| `task_meta.yaml` | GitHub checkout: `tasks/<domain>/<task>/task_meta.yaml` | Catalog identity, lifecycle, runtime packages, prompt mapping, input contract, and declared output files | **Yes in the current runner.** `asibench run` loads it to resolve the task, prepare its runtime, and collect the declared outputs. It is not an answer key. |
| `task_eval.yaml` / `custom_scorer.py` | Same public GitHub task directory | Auditable scoring rules and optional task-specific scorer implementation; formal-task files exclude generation settings and reference content | **No.** Produce-only runs do not execute scoring. Official scoring supplies private references server-side. |
| `framework_task_info.json` | Private scoring dataset: `tasks/<instance-id>/framework_task_info.json` | Frozen scoring-side instance identity, resolved parameters, timeout snapshot, and expected outputs | **No.** Public seed datasets intentionally omit it. Users need only the public prompt, inputs, matching `task_meta.yaml`, and their produced outputs. |

Public scoring files are not required on the participant machine for a
produce-only run. A public result bundle is valid without
`framework_task_info.json`; the scoring side joins it to its own private
instance record and references. The current PyPI wheel contains framework code,
not the GitHub `tasks/` catalog, so the present runner still needs a matching
checkout supplied through `--tasks-dir`.

## The private pieces (you don't see these)

- **The benchmark reference answers** are private and are used only by the
  ASI-Bench website's official scoring workflow.
- **Each graded task's answer generator, generator settings, reference
  specifications, reference answers, and private solver assets** remain on the
  official scoring service. The public `task_eval.yaml` deliberately excludes
  generation, while `custom_scorer.py` exposes scoring logic only.

The fixed-seed datasets give participants only prompts and inputs; private
answers remain on the official scoring service so official scores mean
something. See [How scoring works](how-scoring-works.md).

## The submission portal (website)

Two final confirmations happen on the web:

- **Submitting a run** — after `asibench submit`, you confirm your run in the
  browser to enter the scoring queue (see [Getting started](getting-started.md)).
- **Proposing a task** — the CLI may upload an exact Draft snapshot, but the
  author reviews its files and imported fields and submits the frozen revision
  in the Portal (see [Contribute a task](authoring-a-task.md)).

## In one picture

```
   GitHub: asibench framework  ──pip install──▶  you run your agent locally
                                                        │ outputs
   Hugging Face: seed42/seed31415 ──asibench task pull──▶│
   (prompts + data, no answers)                          ▼
                                          asibench submit ──▶ website (confirm)
                                                                     │
                                          official website scores    │
                                          with private answers ─────▶ leaderboard
```
