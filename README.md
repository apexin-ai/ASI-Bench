<div align="center">

# ASI-Bench: At the Dawn of Artificial Superintelligence

### Evaluating General Intelligence, Innovation, and Autonomous Execution in Scientific Research

<p>
  <a href="https://asibench.apexin.ai/"><img src="https://img.shields.io/badge/Website-ASI--Bench-2563EB?style=for-the-badge&amp;logo=googlechrome&amp;logoColor=white" alt="Website"></a>&nbsp;
  <a href="https://github.com/apexin-ai/ASI-Bench"><img src="https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&amp;logo=github" alt="GitHub"></a>&nbsp;
  <a href="https://github.com/apexin-ai/ASI-Bench/actions/workflows/ci.yml"><img src="https://github.com/apexin-ai/ASI-Bench/actions/workflows/ci.yml/badge.svg" alt="CI"></a>&nbsp;
  <a href="https://huggingface.co/datasets/Apexintelligence-AI/ASI-Bench-seed42"><img src="https://img.shields.io/badge/Data-Hugging%20Face-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=black" alt="Hugging Face"></a>&nbsp;
  <a href="https://asibench.apexin.ai/leaderboard"><img src="https://img.shields.io/badge/Leaderboard-View%20Results-7C3AED?style=for-the-badge&amp;logo=googleanalytics&amp;logoColor=white" alt="Leaderboard"></a>
</p>

<p><sub>
  <strong>Core Authors</strong> &nbsp;
  Junwei Zhou<sup>5,†</sup>, Zhen Sun<sup>1,†</sup>, Binyu Li<sup>1</sup>, Jiangyu Zhou<sup>1</sup>, Yuexi Pan<sup>1</sup>, Hengyu Wang<sup>1</sup>, Honghe Ren<sup>1</sup>, Xiaohan Jia<sup>1</sup>, Xueyang Zhou<sup>1</sup>, Xiaoyu Cao<sup>1</sup>, Yongchao Chen<sup>1,*</sup><br>
  <sup>†</sup> Equal contribution &nbsp;·&nbsp; <sup>*</sup> Corresponding author<br>
  <strong>Contributors</strong> &nbsp;
  Yuanning Feng<sup>1</sup>, Junhao Wu<sup>1</sup>, Cheng Zhang<sup>13</sup>, Sijia Chen<sup>10</sup>, Haoyu Xue<sup>1</sup>, Chengsong You<sup>1</sup>, Huan Wang<sup>1</sup>, Koutian Wu<sup>13</sup>, Peigan Gao<sup>9</sup>, Jiakun Wu<sup>1</sup>, Wenzhe Li<sup>1</sup>, Ergan Shang<sup>4</sup>, Qingyuan Zheng<sup>1</sup>, Jingjing Zhou<sup>1</sup>, Ruixuan Jia<sup>1</sup>, Yan Xu<sup>2</sup>, Hongrui Zhang<sup>7</sup>, Xiao-Han Ma<sup>9</sup>, Zhengxiang Cheng<sup>1</sup>, Yuexing Hao<sup>2</sup>, Liting Mai<sup>6</sup>, Xianglin Ji<sup>2</sup>, Wenjun Zhang<sup>8</sup>, Zhuofan Chen<sup>1</sup>, Yixiao Huang<sup>1</sup>, Chi Wang<sup>12</sup>, Wenyue Hua<sup>11</sup>, Yilun Hao<sup>2</sup>, Yuantao Zhai<sup>1</sup>, Ziyan Zhao<sup>1</sup>, Jingyan Xie<sup>3</sup><br>
  <sup>1</sup> Tsinghua University &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>2</sup> Massachusetts Institute of Technology &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>3</sup> Harvard University &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>4</sup> Carnegie Mellon University &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>5</sup> University of Michigan &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>6</sup> University of Illinois Urbana–Champaign &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>7</sup> Boston University &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>8</sup> University of Queensland &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>9</sup> University of Science and Technology of China &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>10</sup> Flatiron Institute &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>11</sup> Microsoft Research &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>12</sup> AG2 AI &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>13</sup> Independent Researcher
</sub></p>

</div>

<p align="center">
  <img src="docs/assets/asi-bench-overview.png" alt="ASI-Bench overview: 60 project-level tasks across scientific domains and benchmark results across difficulty levels" width="100%">
</p>

<p align="center"><em><strong>Figure 1:</strong> Overview of ASI-Bench. Left: B3 performance across agents. Right: scores from B1 to B4, where B1 provides full methods, B2 only the method name, B3 only the research goal and data, and B4 further adds distractors.</em></p>

## Overview

ASI-Bench is the first benchmark to jointly evaluate **general intelligence,
innovation, and autonomous execution**, and the first to progressively withdraw
human methodological guidance within the same research project. It tests how
far an AI agent can independently select methods, conduct end-to-end research,
and produce verifiable scientific results instead of merely following a
human-specified procedure.

| Benchmark scope | Description |
|---|---|
| Tasks | 60 project-level tasks |
| Scientific coverage | 11 domains |
| Information levels | Four matched B1–B4 conditions with progressively less methodological guidance |
| Evaluation | Expert cross-review, AI-assisted auditing, sandbox execution, and scorer validation |
| Evaluated systems | 18 agent–model configurations |

The scientific objective, input data, required artifacts, and scoring criteria
remain fixed across four matched guidance conditions:

| Level | Information provided | What the agent must do |
|---|---|---|
| B1 | Scientific background, method, equations, and full procedure | Implement and execute the prescribed approach |
| B2 | Intended method and relevant constraints, without the full procedure | Turn the method into a working research workflow |
| B3 | Objective, data, constraints, and required outputs only | Select the method and construct and validate the workflow independently |
| B4 | B3 plus factually correct but non-essential information | Conduct the same autonomous research while resisting distraction |

These are long-horizon investigations rather than isolated questions. Across
the 60 tasks, complete research trajectories involve more than 2,600 interaction
turns and 2,400 execution steps, spanning over 35 hours of agent execution.

The benchmark was distilled from more than 1,300 candidate research ideas
through five review rounds, over 1,100 review assignments, more than 2,000 task
revisions, and over 1,500 sandbox runs. Its construction and validation involved
more than 31,000 human-hours.

### Resources

| Resource | Link |
|---|---|
| Project homepage & documentation | [asibench.apexin.ai](https://asibench.apexin.ai/) |
| Source code | [apexin-ai/ASI-Bench](https://github.com/apexin-ai/ASI-Bench) |
| Leaderboard | [View official results](https://asibench.apexin.ai/leaderboard) |
| Dataset · seed42 | [Apexintelligence-AI/ASI-Bench-seed42](https://huggingface.co/datasets/Apexintelligence-AI/ASI-Bench-seed42) |
| Dataset · seed31415 | [Apexintelligence-AI/ASI-Bench-seed31415](https://huggingface.co/datasets/Apexintelligence-AI/ASI-Bench-seed31415) |
| Python package | [asibench on PyPI](https://pypi.org/project/asibench/) |

## Quick start

### 1. Install ASI-Bench

```bash
pip install asibench
asibench --help
```

**Requirements:** Python 3.11 or later. The default installation is lightweight
and includes everything needed to download public benchmark instances. For
local agent execution or task authoring, install the optional full stack:

`asibench` is the canonical CLI. The legacy `ai4sci-bench` command remains only
as a backwards-compatibility alias.

```bash
pip install 'asibench[full]'
```

### 2. Download a benchmark set

Each fixed-seed dataset contains its own prompts and input data. `seed31415`
also publishes references for local scoring; `seed42` keeps references private.
Choose one seed for each run:

```bash
mkdir asi-bench-run
cd asi-bench-run

asibench task pull \
  --repo seed31415 \
  --output-dir hf_instances_seed31415/
```

### 3. Run an agent

```bash
asibench run \
  --instances-dir hf_instances_seed31415/ \
  --agent-cmd 'python my_agent.py --workspace {workspace}' \
  --sandbox linux_ns \
  --output-dir out_seed31415/
```

The command above evaluates all four prompt levels (`b1,b2,b3,b4`). Use
`--prompt-levels` only when intentionally running a subset.

### 4. Score locally or submit

`seed31415` can be scored locally with its public references and the scoring
contracts in this GitHub checkout:

```bash
asibench score \
  --repo seed31415 \
  --results-dir out_seed31415/ \
  --instances-dir hf_instances_seed31415/ \
  --tasks-dir /path/to/ASI-Bench/tasks/
```

The command writes a separate `local_score_seed31415.json` and never overwrites
the produce-only result. Local scores are reproducible but non-official.

For `seed42`, repeat the pull and run with separate directories, then submit for
private-reference scoring:

```bash
asibench login
asibench submit --results-dir out_seed42/ \
  --benchmark-repo Apexintelligence-AI/ASI-Bench-seed42
```

`submit` creates an authenticated draft on the ASI-Bench website. Review the
completeness summary and confirm the draft to enter the official scoring queue.
Only seed42 result directories are accepted: the CLI checks every instance ID
before building a bundle or authenticating, and rejects seed31415, unknown, or
mixed-seed results. If `--benchmark-repo` is supplied, it must identify the
official seed42 dataset.
`asibench score --repo seed42` is rejected before reading local inputs because
seed42 GT is not public. Local benchmark runs never calculate official scores.

### Run configuration

| Setting | Behavior |
|---|---|
| Dataset seed | `seed31415` supports public local scoring; `seed42` is private-reference and Portal-scored |
| Downloaded instances | `--instances-dir` is read-only; run-specific framework metadata is written under the output directory |
| Prompt levels | All four levels run by default; select a subset with `--prompt-levels` |
| Timeout | `--timeout` defaults to 10,800 seconds and applies uniformly to every task |
| Produce-only reports | Unscored placeholders are never displayed as `0.0`; all-unscored per-task score tables are omitted |
| Submission | `submit` uploads a draft by default; `--no-upload` creates a local bundle only |
| Local scoring | `score --repo seed31415` writes a separate non-official JSON report; seed42 is rejected |
| Custom agents | `--agent-cmd` supports the `none` and `linux_ns` sandboxes |
| Built-in agents | Use `--agent` with `--agent-config`; compatible adapters can use Docker-based `os` isolation |

> **Platform note:** The examples use Linux Bash, and the `linux_ns` sandbox
> requires Linux. Windows users should run it through WSL2. PowerShell and
> Command Prompt use different line-continuation and JSON-escaping syntax.

See [Getting Started](docs/guide/getting-started.md) for agent configuration,
sandbox selection, and platform-specific commands.

### Package and CLI compatibility

- `asibench` is the only published Python distribution and the canonical CLI.
- Existing Python integrations may continue to import `ai4sci_bench`.
- `ai4sci-bench` remains available as a legacy CLI alias.
- Benchmark instances are downloaded separately with `asibench task pull`.
- Runtime and declared-output metadata come from `tasks/` or `--tasks-dir`.
- Retired internal orchestration commands are not part of the public CLI. Use
  `run`, `report`, `batch-report`, `review`, and the Portal workflows.

## Built-in CLI agents — installation

Install the CLI agent you want to evaluate, then select its ASI-Bench adapter
with `--agent`.

- **New model, existing harness.** When a supported harness such as Codex CLI,
  Claude Code, or Kimi Code adds a model, select the new model through
  `--agent-config`. If the model is served through a third-party endpoint and
  the adapter supports that route, also set `api_base`, `api_key`, and
  `api_protocol` (`openai` or `anthropic`). No ASI-Bench code change is normally
  needed while the harness accepts the model and the endpoint remains protocol
  compatible.
- **New agent harness.** A new CLI or runtime can be used immediately through
  `--agent-cmd` when it can read the prepared workspace and write result files;
  this generic mode supports `none` and `linux_ns`. First-class built-in
  integration—its own `--agent` name, harness-specific authentication and
  configuration, proxy wiring, and Docker-based `os` support—requires ASI-Bench
  maintainers to implement and test a new adapter.

| Agent | Install command | Notes |
|-------|----------------|-------|
| Claude Code | `npm install -g @anthropic-ai/claude-code` | Requires Anthropic API key or local login (`claude /login`) |
| Codex CLI | `npm install -g @openai/codex` | Requires OpenAI API key |
| Kimi Code | `npm install -g @moonshot-ai/kimi-code` | **Use the npm package.** The PyPI package (`kimi-cli`) has different CLI arguments and is not compatible with this adapter. Requires Moonshot API key or local login (`kimi /login`) |
| CodeWhale | See [CodeWhale docs](https://github.com/codewhale-ai/codewhale) | Requires DeepSeek API key by default |
| AntiGravity | `curl -fsSL https://antigravity.google/cli/install.sh \| bash` | Experimental |

Example with Kimi Code:

```bash
asibench run --agent kimi_code_cli \
  --agent-config '{"model": "kimi-k2.7"}' \
  --output-dir out/
```

Framework-managed model proxies use text-only input by default. For endpoints
that accept images, set `"supports_image_input": true` in `--agent-config`.
Without this flag, image attachments are replaced with a short notice before
the next model request. Direct endpoints outside the framework proxy remain the
responsibility of the connected agent.

## Contribute a task

We welcome project-level scientific tasks from the research community. Authors
can draft a task locally or in the Portal, validate it, and submit a frozen
revision for review.

### Quick start for task authors

```bash
# 1. Create a task scaffold
asibench task create --domain physics --name my_new_task

# 2. Edit the generated files in tasks/physics/my_new_task/:
#    - task_meta.yaml   — task metadata (public)
#    - task_eval.yaml   — evaluation and generation config during authoring
#    - task_submission.yaml — Portal-only author evidence (private; never exported)
#    - generate_gt.py   — ground-truth generator (private)
#    - custom_scorer.py — custom scoring logic, if needed
#    - prompt_b1.md … prompt_b4.md — local starter prompts for the guidance gradient

# 3. Required pre-submit validation
asibench validate --pre-submit tasks/physics/my_new_task/

# 4. Required same-model trial across B1, B2, B3, and B4
asibench difficulty-check --task physics.my_new_task

# 5. Upload an exact Draft snapshot, then review and submit it in the Portal
asibench task submit --task-dir tasks/physics/my_new_task/
```

### Submission requirements

- A complete task package must define the scientific objective, B1–B4 prompts,
  agent-visible inputs and required artifacts, reproducible reference generation,
  evaluation gates and weighted scorers, runtime dependencies, and local-testing
  evidence.
- Complete `validate --pre-submit` and a same-model `difficulty-check` across
  B1–B4 before submission.
- Use lowercase letters, digits, and underscores for task IDs. The canonical
  form is `domain.task_name`.
- `asibench task submit --task-dir ...` exact-syncs the local files to an
  owner-only Portal Draft. It verifies the returned relative paths and SHA-256
  hashes, and prints the recoverable Draft URL if synchronization fails. It
  never sends the Draft for review by itself.
- On first use, `asibench login` opens **Settings → CLI tokens**. Create a PAT,
  copy it, and paste it into the hidden prompt; the CLI validates and saves it
  in `~/.asibench/credentials` with mode `0600`. Later submissions reuse it.
- Set `ASIBENCH_NO_BROWSER=1` to print URLs without opening a browser. For CI,
  set `ASIBENCH_SUBMIT_TOKEN`; tokens are intentionally not accepted as command
  arguments so they do not enter shell history.
- The Portal requires `local_testing_done` and one finite score for each level
  before freezing a revision.
- Authors do not create repository pull requests. Administrators publish
  accepted tasks after review.

The Portal also provides a 15-step Guided Flow for constructing a Task entirely
in the browser:
[asibench.apexin.ai/submit/proposals/new](https://asibench.apexin.ai/submit/proposals/new).
Use `--endpoint` or `ASIBENCH_SUBMIT_ENDPOINT` to select another Portal.

### Runnable samples from the official benchmark

The following `final` tasks can be used to verify the runner workflow:

| Task | Domain | Description |
|------|--------|-------------|
| `astronomy.nbody_close_encounters` | Astronomy | Close-encounter few-body scattering |
| `math.homotopy_poly_roots` | Math | Isolated complex roots via homotopy continuation |

Pull and run one in produce-only mode:

```bash
asibench task pull \
  --repo seed42 \
  --tasks astronomy.nbody_close_encounters \
  --output-dir example_instances/

asibench run \
  --instances-dir example_instances/ \
  --tasks astronomy.nbody_close_encounters \
  --agent-cmd 'python my_agent.py --workspace {workspace}' \
  --sandbox linux_ns \
  --output-dir example_results/

asibench submit --results-dir example_results/
```

For official benchmark tasks, GitHub publishes the catalog metadata, scoring
configuration, and scorer implementations for auditability. Prompts and input
data come from Hugging Face. Ground-truth generators, generator settings,
reference specifications, and private solver assets remain exclusively on the
scoring service and are not present in this repository. seed42 reference
answers are private. seed31415 reference answers are intentionally public on
Hugging Face, so its GitHub scorers can produce reproducible local,
non-official scores.

Five opt-in `sample` tasks are fully public examples. Their B1–B4 prompts,
ground-truth generators, and scorer implementations or configurations are
available under `tasks/`:

- `robotics.minimum_snap_trajectory_conditioning`
- `chemistry.bsse_counterpoise_cbs_extrapolation`
- `materials.phonon_dispersion`
- `medicine.ethics_disclosure_diagnosis_1670`
- `medicine.jama_id0014_malignant_an`

The examples were synchronized from
[`apexin-ai/Agent-AI4Sci-Bench`](https://github.com/apexin-ai/Agent-AI4Sci-Bench/tree/2ba9258442bf53ad6c4911957234e03e767476ad/tasks)
at revision `2ba9258442bf53ad6c4911957234e03e767476ad`.

### Dataset layout

| Item | Contract |
|---|---|
| Official tasks | 60 `final` task definitions shared by both fixed-seed datasets |
| Public examples | 5 opt-in `sample` tasks with complete prompts and scoring assets |
| Instance layout | `<output-dir>/<instance-id>/` after `asibench task pull` |
| Default selection | Unfiltered official runs exclude `sample` tasks |
| Including samples | Use `--include-sample` or name a sample explicitly with `--tasks` |

Pull the two fixed-seed datasets separately:

```bash
asibench task pull --repo seed42 --output-dir hf_instances_seed42/
asibench task pull --repo seed31415 --output-dir hf_instances_seed31415/
```

`seed42` and `seed31415` are the only official Hugging Face contracts, and
`--repo` is required. They share task schemas and declared outputs but contain
different generated inputs. Matching schemas live in
`tasks/**/task_meta.yaml`; matching public scoring contracts live in
`task_eval.yaml` and optional `custom_scorer.py` files. seed31415 additionally
contains public `reference/` directories. Private manifests, ground-truth
generators, generator settings, seed42 references, and solver assets stay on
the scoring side.

The task workflow uses the grouped `asibench task create`, `task pull`, and
`task submit` commands. See [Contribute a Task](docs/guide/authoring-a-task.md)
for the complete authoring guide.

## How scoring works (and stays fair)

- **Public execution framework:** agent adapters, sandboxes, instance loading,
  output collection, and submission packaging are auditable in this repository.
- **Public local scoring:** seed31415 publishes references on Hugging Face and
  uses the GitHub scoring configuration/custom scorers through `asibench score`.
- **Private official scoring:** seed42 never publishes references;
  `asibench submit` accepts only seed42 outputs and sends them to the website
  for authenticated scoring.
- **Score status:** seed31415 local reports are explicitly non-official;
  its results cannot be packaged by `asibench submit`, and self-reported scores
  are not accepted as leaderboard results.

## Sandboxing & reproducibility

- Runs support `task`, `os`, and `linux_ns` sandboxes with pinned per-task
  runtimes.
- Every run records agent, model, effort, sandbox, and framework provenance.
- Source checkouts are installed editable into task environments; wheel-based
  runs install the matching `asibench` version.
- Mutable caches live under `~/.asibench/`, never under `site-packages`.
- Namespace and Docker prerequisites are checked before execution. Missing
  prerequisites produce a non-zero CLI exit.

## Documentation

Installation, execution, submission, task contribution, catalog, and
leaderboard documentation are available on the
[ASI-Bench website](https://asibench.apexin.ai/).

## License

See [LICENSE](LICENSE).
