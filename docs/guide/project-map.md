# Where Things Live

ASI-Bench is spread across a few places. As an external user you interact with
the code on GitHub and fixed-seed datasets on Hugging Face. seed31415 answers
are public for local scoring; seed42 answers remain private.

## The public pieces

| Where | What it holds | You use it to… |
|---|---|---|
| **GitHub — the `asibench` repo** | The produce-only runner and public task metadata; official prompts are not tracked here | `pip install asibench`, run the benchmark, and package outputs |
| **Hugging Face — seed42** | One official fixed-seed task set with prompts and input data | `asibench task pull --repo seed42` |
| **Hugging Face — seed31415** | Prompts, input data, and public references | `task pull`, then `asibench score --repo seed31415` |

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
| `task_eval.yaml` / `custom_scorer.py` | Same public GitHub task directory | Auditable scoring rules and optional task-specific scorer implementation; formal-task files exclude generation settings and reference content | **For local scoring only.** seed31415 needs a GitHub checkout supplied through `--tasks-dir`; seed42 scoring uses the website. |
| `framework_task_info.json` | Optional pulled/run snapshot | Frozen framework identity, resolved parameters, timeout snapshot, and expected outputs; it is not an answer key | **No.** Public runs and submission bundles remain valid when it is absent. |

Public scoring files are not required for a produce-only run. A result bundle
is valid without `framework_task_info.json`; the scoring side joins it to its own private
instance record and references. The current PyPI wheel contains framework code,
not the GitHub `tasks/` catalog, so the present runner still needs a matching
checkout supplied through `--tasks-dir`.

## Public and private reference pieces

- **seed31415 reference answers** are public in its Hugging Face instance tree.
- **seed42 reference answers** are private and used only by the ASI-Bench website.
- **Each graded task's answer generator, generator settings, reference
  specifications, and private solver assets** remain on the official scoring
  service. The public `task_eval.yaml` deliberately excludes
  generation, while `custom_scorer.py` exposes scoring logic only.

Both datasets provide prompts and inputs. Only seed31415 provides references;
seed42 answers remain on the scoring service. See
[How scoring works](how-scoring-works.md).

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
   Hugging Face: seed31415 ──task pull──▶ outputs + public refs ──score──▶ local report
   Hugging Face: seed42    ──task pull──▶ outputs
                                                        ▼
                                          asibench submit ──▶ website (confirm)
                                                                     │
                                          official website scores    │
                                          with private answers ─────▶ leaderboard
```
