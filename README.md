<div align="center">

# ASI-Bench: AI Scientist’s Last Exam

### A Project-Level Benchmark for Evaluating LLM Agents on AI for Science

<p>
  <a href="https://asibench.apexin.ai/"><img src="https://img.shields.io/badge/Website-ASI--Bench-2563EB?style=for-the-badge&amp;logo=googlechrome&amp;logoColor=white" alt="Website"></a>&nbsp;
  <a href="https://github.com/apexin-ai/ASI-Bench"><img src="https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&amp;logo=github" alt="GitHub"></a>&nbsp;
  <a href="https://github.com/apexin-ai/ASI-Bench/actions/workflows/ci.yml"><img src="https://github.com/apexin-ai/ASI-Bench/actions/workflows/ci.yml/badge.svg" alt="CI"></a>&nbsp;
  <a href="https://huggingface.co/datasets/Apexintelligence-AI/ASI-Bench-seed42"><img src="https://img.shields.io/badge/Data-Hugging%20Face-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=black" alt="Hugging Face"></a>&nbsp;
  <a href="https://asibench.apexin.ai/leaderboard"><img src="https://img.shields.io/badge/Leaderboard-View%20Results-7C3AED?style=for-the-badge&amp;logo=googleanalytics&amp;logoColor=white" alt="Leaderboard"></a>
</p>

<p><sub>
  <strong>Core Authors</strong> &nbsp;
  Junwei Zhou<sup>3,†</sup>, Zhen Sun<sup>1,2,†</sup>, Binyu Li<sup>1,2</sup>, Jiangyu Zhou<sup>1,2</sup>, Yuexi Pan<sup>1,2</sup>, Hengyu Wang<sup>1,2</sup>, Honghe Ren<sup>1</sup>, Xiaohan Jia<sup>1,2</sup>, Xueyang Zhou<sup>1,2</sup>, Yongchao Chen<sup>1,2,*</sup><br>
  <sup>†</sup> Equal contribution &nbsp;·&nbsp; <sup>*</sup> Corresponding author<br>
  <strong>Contributors</strong> &nbsp;
  Yuanning Feng<sup>1,2</sup>, Junhao Wu<sup>1,2</sup>, Xiaoyu Cao<sup>1,2</sup>, Cheng Zhang<sup>3</sup>, Sijia Chen<sup>5</sup>, Haoyu Xue<sup>1,2</sup>, Chengsong You<sup>1,2</sup>, Huan Wang<sup>1,2</sup>, Peigan Gao<sup>4</sup>, Jiakun Wu<sup>1</sup>, Koutian Wu<sup>3</sup>, Wenzhe Li<sup>1</sup>, Ergan Shang<sup>6</sup>, Jingjing Zhou<sup>1</sup>, Ruixuan Jia<sup>1,2</sup>, Qingyuan Zheng<sup>1</sup>, Yan Xu<sup>7</sup>, Hongrui Zhang<sup>8</sup>, Xiao-Han Ma<sup>4</sup>, Zhengxiang Cheng<sup>1,2</sup>, Yuexing Hao<sup>7</sup>, Liting Mai<sup>9</sup>, Xianglin Ji<sup>7</sup>, Wenjun Zhang<sup>10</sup>, Zhuofan Chen<sup>1,2</sup>, Yixiao Huang<sup>1</sup>, Chi Wang<sup>11</sup>, Wenyue Hua<sup>12</sup>, Yilun Hao<sup>7</sup>, Yuantao Zhai<sup>2</sup><br>
  <sup>1</sup> Tsinghua University &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>2</sup> Apex Intelligence &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>3</sup> Independent Researcher &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>4</sup> University of Science and Technology of China &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>5</sup> Flatiron Institute &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>6</sup> Carnegie Mellon University &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>7</sup> Massachusetts Institute of Technology &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>8</sup> Boston University &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>9</sup> University of Illinois Urbana–Champaign &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>10</sup> University of Queensland &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>11</sup> AG2 AI &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>12</sup> Microsoft Research
</sub></p>

</div>

<p align="center">
  <img src="docs/assets/asi-bench-overview.png" alt="ASI-Bench overview: 60 project-level tasks across scientific domains and benchmark results across difficulty levels" width="100%">
</p>

<p align="center"><em><strong>Figure 1:</strong> Overview of ASI-Bench. Left: B3 performance across agents. Right: scores from B1 to B4.</em></p>

## Overview

ASI-Bench evaluates **scientific autonomy**: whether an AI agent can translate a
scientific objective into an executable and independently validated workflow,
rather than merely follow prescribed procedures.

| Benchmark scope | Description |
|---|---|
| Tasks | 60 project-level tasks |
| Scientific coverage | 11 domains |
| Information levels | B1–B4, with progressively less methodological guidance |
| Evaluation | Expert cross-review, execution testing, and reliable scoring |
| Evaluated systems | 18 agent–model configurations |

Across all evaluated systems, the average scores are **50.92 at B1**, **29.64 at
B2**, **27.17 at B3**, and **27.33 at B4** as concrete procedural guidance is
progressively removed. The sharp drop after B1 and persistently lower scores at
B2–B4 show that strong workflow execution does not yet imply independent
scientific problem-solving.

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

Each fixed-seed dataset contains its own prompts and input data. Choose one seed
for each run:

```bash
mkdir asi-bench-run
cd asi-bench-run

asibench task pull \
  --repo seed42 \
  --output-dir hf_instances_seed42/
```

### 3. Run an agent

```bash
asibench run \
  --instances-dir hf_instances_seed42/ \
  --agent-cmd 'python my_agent.py --workspace {workspace}' \
  --sandbox linux_ns \
  --output-dir out_seed42/
```

The command above evaluates all four prompt levels (`b1,b2,b3,b4`). Use
`--prompt-levels` only when intentionally running a subset.

### 4. Submit for official scoring

```bash
asibench login
asibench submit --results-dir out_seed42/
```

`submit` creates an authenticated draft on the ASI-Bench website. Review the
completeness summary and confirm the draft to enter the official scoring queue.
Local benchmark runs do not calculate official scores.

### Run configuration

| Setting | Behavior |
|---|---|
| Dataset seed | `seed42` and `seed31415` are separate runs; use distinct instance and output directories |
| Prompt levels | All four levels run by default; select a subset with `--prompt-levels` |
| Timeout | `--timeout` defaults to 10,800 seconds and applies uniformly to every task |
| Submission | `submit` uploads a draft by default; `--no-upload` creates a local bundle only |
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
| AntiGravity | `curl -fsSL https://antigravity.google/cli/install.sh | bash` | Experimental |

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
#    - task_eval.yaml   — evaluation config: scoring rules, parameter space (private)
#    - generate_gt.py   — ground-truth generator (private)
#    - custom_scorer.py — custom scoring logic, if needed (private)
#    - prompt_b1.md … prompt_b4.md — difficulty-ladder prompts (public)

# 3. Required pre-submit validation
asibench validate --pre-submit tasks/physics/my_new_task/

# 4. Required same-model trial across B1, B2, B3, and B4
asibench difficulty-check --task physics.my_new_task

# 5. Open the Portal task-submission form
asibench task submit
```

### Submission requirements

- Complete `validate --pre-submit` and a same-model `difficulty-check` across
  B1–B4 before submission.
- Use lowercase letters, digits, and underscores for task IDs. The canonical
  form is `domain.task_name`.
- `asibench task submit` opens the Portal form; files are uploaded and reviewed
  in the browser, not from the terminal.
- Set `ASIBENCH_NO_BROWSER=1` to print the form URL without opening a browser.
- The Portal requires `local_testing_done` and one finite score for each level
  before freezing a revision.
- Authors do not create repository pull requests. Administrators publish
  accepted tasks after review.

The default submission form is
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

For official benchmark tasks, the GitHub catalog contains metadata only.
Prompts and input data come from Hugging Face, while reference answers and
scorers remain on the private scoring service.

The framework-level `tasks/_template/` directory is the only public authoring
scaffold with local generator and scoring configuration. Formal benchmark task
directories contain `task_meta.yaml` only.

### Dataset layout

| Item | Contract |
|---|---|
| Official tasks | 60 `final` task definitions shared by both fixed-seed datasets |
| Author scaffold | `tasks/_template/` with local starter files |
| Instance layout | `<output-dir>/<instance-id>/` after `asibench task pull` |
| Default selection | Unfiltered runs use the official `final` task catalog |

Pull the two fixed-seed datasets separately:

```bash
asibench task pull --repo seed42 --output-dir hf_instances_seed42/
asibench task pull --repo seed31415 --output-dir hf_instances_seed31415/
```

`seed42` and `seed31415` are the only official Hugging Face contracts, and
`--repo` is required. They share task schemas and declared outputs but contain
different generated inputs. Matching schemas live in
`tasks/**/task_meta.yaml`; private manifests, references, and scorers stay on
the scoring side.

The task workflow uses the grouped `asibench task create`, `task pull`, and
`task submit` commands. See [Contribute a Task](docs/guide/authoring-a-task.md)
for the complete authoring guide.

## How scoring works (and stays fair)

- **Public execution framework:** agent adapters, sandboxes, instance loading,
  output collection, and submission packaging are auditable in this repository.
- **Centralized official scoring:** `asibench login` identifies the submitter,
  and `asibench submit` sends a draft for confirmation and scoring. Self-reported
  scores are not accepted.

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
