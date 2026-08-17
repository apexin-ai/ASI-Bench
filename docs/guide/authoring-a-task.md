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

- **B1** — full background and a complete method description.
- **B2** — the method with some steps left out.
- **B3** — minimal hints.
- **B4** — essentially just the goal.

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

Run the full pre-submit validation, then test the same model with the same run
settings against all four prompt levels:

```bash
# schema, required files, ground-truth generation, and scorer self-check
asibench validate --pre-submit tasks/physics/my_task/

# defaults to B1, B2, B3, and B4
asibench difficulty-check --task physics.my_task
```

`--pre-submit` runs your generator and feeds its reference output through your
own `task_eval.yaml`. A failure means the generator and scorer disagree and must
be fixed before submission.

In the Portal's **Local testing** step, confirm that you ran the same model and
settings for B1–B4, select the provider/model version, and enter all four finite
scores (0–100). `local_testing_done` plus one score for each level is a submission
requirement, not an optional note.

!!! tip "Determinism"
    Run `generate_gt.py` twice with the same seed and confirm the outputs are
    identical. Seed every random number generator from the `seed` parameter. A task
    that is not reproducible may be rejected or returned for changes.

---

## 4. Submit through the Portal

Task submission is completed only in the Portal web form. The CLI provides a
convenience launcher but does not upload task files or create a draft.

### Open the web form from the CLI

The command defaults to the official Portal. Pass a different Portal URL directly
or configure `ASIBENCH_SUBMIT_ENDPOINT` to override it:

```bash
asibench task submit
```

This opens `https://asibench.apexin.ai/submit/proposals/new` and prints
the URL. It intentionally has no
`--task-dir`, `--token`, `--submit`, or file-synchronization options. This keeps
the detailed metadata, file review, local-test evidence, scoring information,
and author confirmations in one guided web flow. On a headless machine, set
`ASIBENCH_NO_BROWSER=1` and open the printed URL elsewhere.

### Complete the submission in the browser

1. Open the **submission Portal** and sign in.
2. Start **"Propose a Task"** and complete the guided fields.
3. Upload the four prompts, generator, evaluation config, and optional scorer.
4. Review the completeness summary and the full file list.
5. Record the model identity and all four B1–B4 local-test scores.
6. Click **Submit** to freeze the revision for review.

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
