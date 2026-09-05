# Contribute a Task

ASI-Bench grows through community-contributed scientific tasks. This guide is for
external researchers who want to propose a new task: what a good task looks like,
what files it is made of, how to check it locally, and how to submit it through the
submission portal.

You do **not** need write access to any repository and you do not create a pull
request. Author with the public `asibench` tools or the Portal, then submit a
revision for assigned human review.

---

## 1. What makes a good ASI-Bench task

A strong task is:

- **Scientifically meaningful** — it tests a real method or result from your field
  (a simulation, an inference problem, a numerical method), not a toy puzzle.
- **Hard for AI agents, not just for humans** — a capable model with the full
  method description should still find it non-trivial. The task ships a **difficulty
  ladder** of prompts (B1 → B4) that reveals less and less.
- **Deterministic and reproducible** — given the same parameters and random seed,
  the reference answer is always the same, so scoring is stable.
- **Objectively scorable with tolerances** — outputs are compared to a reference
  numerically (with tolerances) or by structural checks, not by opinion.
- **Self-contained** — the agent is given input data + a prompt and must produce
  the declared output files; no internet or task-specific "solver" library required.

---

## 2. Anatomy of a task

A task is a small directory. Start it with `asibench task create`. A source checkout
contains a metadata/generator template under `tasks/_template/`; the CLI generates
fresh B1–B4 starter prompt files in the new task directory. The installed package
generates all starter files directly and does not install that repository directory:

| File | What it is | Public? |
|---|---|---|
| `task_meta.yaml` | Public metadata: id, name, domain, difficulty, runtime, and the **names/types** of inputs and outputs | Public |
| `prompt_b1.md … prompt_b4.md` | The difficulty ladder — B1 gives the full method, B4 gives almost nothing | Public |
| `generate_gt.py` | Deterministic generator: given params + seed, writes the input data **and** the reference answer | Private |
| `task_eval.yaml` | During authoring: scoring and generation settings. After acceptance: only the scoring contract is published | Split on acceptance |
| `custom_scorer.py` *(optional)* | A bespoke scorer, if the generic scorers are not enough | Public after acceptance |
| `reference_specs.md` | A short description of what a correct answer looks like | Private |

You author all of these files. When your task is accepted, the **metadata,
prompts, input data, scoring-only configuration, and scorer implementation** are
published so runners can attempt and audit it. The **GT generator, generation
settings, reference specifications, reference answers, and private solver
assets stay private**. You do not manage this split yourself; the portal and
maintainers handle it.

### The difficulty ladder (B1–B4)

Every task provides four prompt levels of decreasing guidance:

- **B1** — scientific background, method, equations, and full procedure.
- **B2** — the intended method and constraints, without the full procedure.
- **B3** — objective, data, constraints, and required outputs only.
- **B4** — B3 plus factually correct but non-essential information.

A good task is one where scores drop meaningfully as the prompt gives less away.
Make sure a hint that appears at B1 does **not** leak into B3/B4.

---

## 3. Complete the required local checks

Everything here uses the public `asibench` package — no private access. You may
create and edit a Portal draft at any time, but the Portal will not freeze and
submit a revision until the required local validation and B1–B4 results are recorded.

```bash
pip install asibench

# scaffold a new task directory from the template
asibench task create --domain physics --name my_task

# ... edit the files under tasks/physics/my_task/ ...
```

Fill in `task_meta.yaml`, the four prompts, `generate_gt.py`, and the evaluation
config, using `tasks/_template/` as a reference.

Run the full pre-submit validation, then explicitly select a multi-turn agent,
model, and run settings for all four prompt levels:

```bash
# schema, required files, ground-truth generation, and scorer self-check
asibench validate --pre-submit tasks/physics/my_task/

# B1, B2, B3, and B4 run by default
asibench difficulty-check --task physics.my_task \
  --agent codex_cli \
  --agent-config '{"model":"gpt-5.6-sol","effort":"medium"}' \
  --sandbox os
```

`--pre-submit` runs your generator and feeds its reference output through your
own `task_eval.yaml`. A failure means the generator and scorer disagree and must
be fixed before submission.

In the Portal's **Local testing** step, confirm that you ran the same model and
settings for B1–B4, select the provider/model version, and enter all four finite
scores (0–100). `local_testing_done` plus one score for each level is a submission
requirement, not an optional note. The Portal's Task contribution policy also
requires this evidence to be recorded from the `os` (Docker) sandbox; evidence
from `none`, `task`, or `linux_ns` cannot unlock submission.

Task contribution trials must run in the Docker OS sandbox. The command defaults
to `--sandbox os` and rejects `none`, `task`, and `linux_ns`. Record
`sandbox: os` on every entry in `task_submission.yaml.local_test_results`; the
CLI validates this evidence before contacting the Portal.

`difficulty-check` intentionally has no default agent or model. Each repeated
`--agent` requires a position-matched `--agent-config` containing `model`, and
formal evidence must include at least one supported multi-turn harness such as
`codex_cli`, `claude_code_cli`, or `kimi_code_cli`. `direct_llm` remains useful
as an optional single-turn baseline but cannot be the only difficulty agent.
When multiple agents are supplied, every gated agent/B3/B4 mean must remain
strictly below 40. JSON, Markdown, CSV, and persisted score reports record the
agent, effective model, effort, measured agent version, framework version, and
sandbox.

The difficulty gate applies only to the low-guidance levels: every B3 and B4
mean score must be strictly below 40. B1 and B2 are required evidence but have
no score ceiling. The CLI therefore reports them as `RECORDED`, while B3/B4
receive `PASS` or `FAIL`. You may lower `--threshold` for a stricter check, but
the CLI rejects values above 40.

!!! tip "Determinism"
    Run `generate_gt.py` twice with the same seed and confirm the outputs are
    identical. Seed every random number generator from the `seed` parameter. A task
    that is not reproducible may be rejected or returned for changes.

---

## 4. Upload a Draft, then confirm it in the Portal

The CLI can exact-sync a local Task directory into an owner-only Portal Draft.
It deliberately cannot make the final submission: the author must inspect the
exact file list and imported fields in the browser and click **Submit** there.

The command defaults to the official Portal. Pass a different Portal URL or set
`ASIBENCH_SUBMIT_ENDPOINT` to override it:

```bash
asibench task submit --task-dir tasks/physics/my_new_task/
```

On first use, run `asibench login` (or let `task submit` prompt). The CLI opens
Portal Settings, where you manually create and copy a PAT. Paste it into the
hidden terminal prompt; it is validated with the Portal and saved locally with
file mode `0600`. If the clipboard changed, validation fails without saving and
you can paste again. Later runs reuse the saved token. For CI/headless use,
provide `ASIBENCH_SUBMIT_TOKEN`; there is no token command-line flag.

The Task directory must include `task_submission.yaml` for Portal-only author
evidence and local-test results. That file is uploaded and frozen for review but
is never exported to the benchmark Task repository. After exact synchronization,
the CLI opens `/submit/proposals/<id>`. Set `ASIBENCH_NO_BROWSER=1` to print the
URL instead. The CLI verifies both the complete relative-path set and every
SHA-256 hash returned by the Portal; if synchronization fails after Draft
creation, the error includes that Draft's recovery URL.

### Complete the submission in the browser

1. Review the completeness summary and the exact synchronized file list.
2. Inspect and, if necessary, edit the imported metadata and evidence.
3. Click **Submit** to freeze the revision for review.

Files uploaded through the form are private to the author, assigned reviewers,
and administrators.

Task authors do not create repository pull requests. Approval and later
repository publication are separate administrator actions.

---

## 5. What happens after you submit

1. Clicking **Submit** freezes a revision; later edits do not change that snapshot.
2. An administrator assigns a reviewer. The task author is excluded as a reviewer.
3. The assigned reviewer or an administrator can view or download the full revision.
4. They choose **Approve**, **Request Changes**, or **Reject**, with feedback.
5. After **Request Changes**, edit the draft and submit a new revision; prior
   revisions remain available in history.
6. Approval does not publish automatically. Administrators handle later repository
   publication and pull requests separately.

You are notified when the task status or review result changes.

---

## Common pitfalls

- **Non-deterministic generators** — unseeded randomness, or numerics that depend on
  hardware. Seed everything; keep tolerances realistic.
- **Answer leakage** — the reference answer (or a value that trivially yields it)
  appearing in a prompt or in the input data.
- **Prompt-level leakage** — a hint meant for B1 showing up in B3/B4.
- **Requiring a library that just solves the problem** — the point is to test the
  agent's method, not its ability to import a solver. Declare genuinely-needed
  packages in `task_meta.yaml` runtime.
- **Scorer that only accepts an exact match** — real numerical work has floating
  point error; score with sensible tolerances.

---

## Next steps

- [Getting started](getting-started.md) — install and run the benchmark.
- [How scoring works](how-scoring-works.md) — the public/private split.
