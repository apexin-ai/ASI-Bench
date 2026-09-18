# ASI-Bench 适配器使用指南

将 ASI-Bench seed31415 的一个 task 接入 BenchFlow，分四步执行：

```text
下载 instance + task bundle → prepare → Agent 解题并收集输出 → 独立容器评分
```

本文描述当前实现，不保留历史阶段的过期命令。详细设计和验收记录见
`docs/plans/asi-bench-scoring-implementation-plan.md`。

## 1. 支持范围与限制

| 项目 | 当前支持情况 |
| --- | --- |
| 数据集 | Hugging Face `Apexintelligence-AI/ASI-Bench-seed31415`；需要公开 reference |
| sandbox mode | **仅支持 ASI `os` 语义的 Docker 沙箱**，BenchFlow backend 固定为 `docker` |
| 其他沙箱 | 不支持宿主机直接执行、Daytona 或其他后端；CLI 没有 `--sandbox` 切换参数 |
| 运行环境 | `cpu-v1`：2 CPU、4096 MB 内存、`/workspace` 工作目录；不支持 GPU 或要求联网的任务 |
| prompt level | `b1`–`b4`，prepare 时选择；solve 使用 prepared 中记录的 level |
| Agent | 使用 BenchFlow 原生 Agent/ACP 与 rollout；必须显式指定 agent、model；不是所有 Agent 组合都已验收 |
| 模型验收示例 | `claude-agent-acp` + 自定义 OpenAI-compatible 路由，模型参数使用 `vllm/<model>` |
| 评分器 | 当前 evaluator 包含的非 Judge scorer，以及依赖可满足的 task 自定义 scorer；不保证所有任务均可运行 |
| **LLM/VLM Judge** | **不支持**，包括依赖模型调用的 `agent_judge`；不加载完整 Judge 依赖链，不提供 Judge 凭据或网络 |
| 评分性质 | seed31415 本地 **non-official** 评分；不支持 seed42 评分或官方提交 |

任务的 gates/scoring 若依赖未包含的 Judge，可能在导入或注册检查时失败。
**不会自动跳过 Judge，也不会把它替换成零分或视为通过。**

### 网络与权限

- 下载及 Docker **构建阶段**可联网，安装任务声明的依赖。
- 解题阶段保留 CLI Agent 调用模型所需的连接，使用 BenchFlow 原生代理、防火墙及 no-web 策略；
  这不等于给任务开放任意网络。Agent 不以 root 运行，不通过关闭防火墙绕过错误。
- 评分阶段使用独立的断网容器、非 root 用户、只读根文件系统和只读输入挂载。
  评分执行期间不安装依赖，不调用 Agent，不执行 LLM Judge。

### ASI-Bench → BenchFlow 映射（How it maps to BenchFlow）

本适配器复用 BenchFlow 的任务描述、Docker sandbox、Agent/ACP 和底层 Rollout，
在适配层处理 ASI 的输入、输出与评分契约；不是通过 `asibench run` 调用 ASI 原生 Agent adapter。

| ASI-Bench 概念或文件 | BenchFlow / 适配侧对应形式 | 负责位置 |
| --- | --- | --- |
| seed31415 instance + task bundle | raw 资产与 `sources.json`，记录来源 revision 和文件摘要；不直接作为 Agent 工作目录 | `sources.py` |
| Task 元数据、输入/输出契约 | `task_manifest.json`、匿名化 `environment/inputs/task_info.json`，以及 task/sandbox 配置 | `prepare.py`、`solve.py` |
| Instance 的 B1–B4 prompt | prepare 选择一个 level，生成 `task.md` 和 `environment/inputs/prompt.md`；solve 校验后将该 prompt 显式传给 Rollout | `prepare.py`、`solve.py` |
| Agent 可见的实例输入 | `environment/inputs/` 作为 Docker 构建输入，复制到 Agent 容器 `/workspace`；不包含 reference 或评分代码 | `prepare.py` |
| Task runtime 依赖与资源需求 | `environment/Dockerfile`、`runtime-constraints.txt`、`runtime-profile.json` 和 Rollout sandbox 配置 | `prepare.py`、`solve.py` |
| ASI `os` sandbox | BenchFlow `docker` backend；适配层提供运行副本，网络执行仍使用原生代理/防火墙机制 | `solve.py` 的 `ASIDockerPlanes` |
| Agent adapter 执行 | `RolloutConfig` 指定 agent/model，底层 Rollout 调用 BenchFlow 原生 Agent/ACP；适配器不重新实现 Agent harness | `solve.py` |
| Task 声明的输出文件 | 停止容器写入后，从 `/workspace` 导出声明产物，保存为 `<instance>__<level>.outputs/` | `solve.py` |
| 解题结果与执行证据 | ASI 风格目录和结果 JSON，绑定 `benchflow/` 下原生 rollout 结果、日志及轨迹；不是 ASI 原生结果 schema 的完整复制 | `solve.py` |
| ASI gates、scorers、聚合逻辑 | 根据 prepared Dockerfile 重建镜像，在独立 DockerSandbox 评分容器内运行所选 ASI evaluator 代码及适配评分入口 | `scorer.py`、`templates/verifier/score_entry.py`、`templates/verifier/scoring.py` |
| Scorer 的 prediction 工作目录 | 临时只读 `/prediction/` = Agent 输出 + 可信 instance `data/`；reference 独立挂载，不混入 prediction | `scorer.py` |
| 最终分数与 reward | 独立输出 `asibench_score.json`、`reward.txt`、`execution.json`，记录实际评分镜像和代码身份 | `scorer.py`、评分入口 |

**执行边界：**

- solve 使用 `Rollout.create` 和分阶段生命周期：`setup → start → install_agent → connect → execute`，
  随后断开、冻结并导出产物、finalize/cleanup；不通过完整 `SDK.run / Evaluation` 流程编排评分。
- solve 配置 `skip_verify=True`，不执行原生 verifier。prepared 中虽然有 `verifier/`，
  当前评分由独立的 `scorer.score_result` 入口执行，不由 solve 的 verifier 阶段触发。
- 独立评分输出的 `reward.txt` **不会自动回填**原生 RolloutResult，也不会覆盖 produce-only 结果的
  `evaluation_status: pending`；最终评分状态以独立评分报告为准。

## 2. 环境准备

以下命令在 **BenchFlow 仓库或当前适配 worktree 根目录**执行，不是在 ASI-Bench 仓库执行。
需要 `uv`、可访问的 Docker daemon 及 Docker Compose。

```bash
cd /path/to/benchflow
uv sync --extra dev --locked

docker info
docker compose version
```

下载依赖声明在 `benchmarks/asi_bench/requirements.txt`。下面下载命令通过
`uv run --with-requirements` 显式提供，避免误用系统 Python 导致 `ModuleNotFoundError`。

评分需要的框架代码由宿主机在构建评分镜像前从 GitHub 固定 revision 下载，
**不需要本地 ASI-Bench 源码，也不需要传 `--asi-source`**。
`evaluator_files.json` 记录上游 revision、原始文件 SHA-256 和适配方式：
7 个 `copy` 文件原样下载；`core/__init__.py` 本地生成最小导出；
`scorers/__init__.py` 与 `runner/__init__.py` 生成为空文件；
`runner/task_env.py` 使用本地 `templates/verifier/task_env.py`，不重复下载。

原始文件缓存位于 `~/.cache/benchflow/asi_bench/evaluator_sources/<revision>/`。
首次评分需要宿主机能访问 `raw.githubusercontent.com`，有效缓存可离线复用；
缓存摘要不匹配时重新下载，下载摘要不匹配则报错，不继续评分。
每次评分重新组装临时 evaluator，只读挂载进独立、断网的评分容器；
不在宿主机导入这组评分代码，也不缓存组装后的 runtime。
许可证和来源维护说明见本文末尾。

本次分发方式变更会改变 evaluator 身份摘要。旧 example 是历史验证快照，
不会自动迁移；新验收请重新 prepare，再使用匹配的新解题结果评分。
任务专属的 `custom_scorer.py` 等仍来自下载的 task bundle，不包含在框架副本里。

## 3. 推荐的端到端命令

### 3.1 选择任务与独立目录

以下以 `math.mpsc_safety_filter`、B1 为命令示例。换任务只需修改变量，
但先确认它符合上述运行环境和非 Judge 限制。

```bash
export TASK_ID=math.mpsc_safety_filter
export INSTANCE_ID="${TASK_ID}__seed31415"
export LEVEL=b1
export ACCEPT_ROOT="$HOME/asi-bench-acceptance/$(date +%Y%m%d-%H%M%S)"
export RAW_DIR="$ACCEPT_ROOT/raw"
export RUN_DIR="$ACCEPT_ROOT/run"
export SCORE_DIR="$ACCEPT_ROOT/score"
mkdir -p "$ACCEPT_ROOT"
```

`ACCEPT_ROOT` 只是便于组织验收文件的 shell 变量，不是程序参数。
raw、prepared、run 和 score 应为同级或互不重叠的目录。

### 3.2 下载 instance 和 task bundle

```bash
uv run --with-requirements benchmarks/asi_bench/requirements.txt \
  python -m benchmarks.asi_bench.sources \
  --task-id "$TASK_ID" \
  --output-dir "$RAW_DIR"
```

- `--output-dir` 是本次下载的**完整目标目录**，必须不存在，不是已存在的父目录。
- 不指定时默认为 `~/.cache/benchflow/asi_bench/downloads/<task-id>/`；设置
  `XDG_CACHE_HOME` 时使用该缓存根。HF 缓存默认位于对应根下的 `benchflow/asi_bench/hf/`。
- 可用 `--hf-revision`、`--asi-revision` 固定来源版本，默认都是 `main`，解析后的 commit 写入 `sources.json`。
- `HF_TOKEN`、`GITHUB_TOKEN` 可通过宿主机环境提供；不要放进命令行参数或提交到仓库。
- 已下载完成时直接复用 raw 进入 prepare；不要再次向同一个目录下载。

### 3.3 转换为 prepared task

```bash
uv run python -m benchmarks.asi_bench.prepare \
  --raw-dir "$RAW_DIR" \
  --task-id "$TASK_ID" \
  --instance-id "$INSTANCE_ID" \
  --level "$LEVEL" \
  --output-dir "$ACCEPT_ROOT/prepared"
```

将返回 JSON 中的 `output_dir` 设置为 `PREPARED_DIR`，不要只传 prepared 父目录：

```bash
export PREPARED_DIR="$ACCEPT_ROOT/prepared/$TASK_ID/<实际返回的task-id>"
```

prepare 生成 `task.md`、环境 Dockerfile、Agent 输入和 verifier 入口。
原始 task bundle/reference 保留在 raw 中，prepared 通过 manifest 引用它们，**不要移动或删除 raw**。
prepared 文件不可手工修改；相同输入产生的目录可在完整性检查通过后复用。

### 3.4 配置模型

在仓库之外创建私有 env 文件，例如 `$HOME/.config/asi-bench/model.env`：

```dotenv
ASI_MODEL_BASE_URL=https://your-provider.example/v1
ASI_MODEL_API_KEY=replace-with-your-key
```

```bash
chmod 600 "$HOME/.config/asi-bench/model.env"
export MODEL_ENV_FILE="$HOME/.config/asi-bench/model.env"
```

这些变量配置的是 **解题模型**，不提供 Judge 支持。不要将 key 写入 `--model`、
命令行、README 或 Git。适配器元数据保存 endpoint 和变量名，不保存 key 值。

使用自定义 OpenAI-compatible endpoint 时，下面示例采用 `vllm/<model>` 路由。
裸模型名可能选择其他原生 provider 并要求其凭据（例如 `OPENAI_API_KEY`），
不能假定所有模型名都会使用 env 文件中的自定义 endpoint。

### 3.5 调用 Agent 解题

```bash
uv run python -m benchmarks.asi_bench.solve \
  --prepared-dir "$PREPARED_DIR" \
  --run-dir "$RUN_DIR" \
  --model-env-file "$MODEL_ENV_FILE" \
  --agent claude-agent-acp \
  --model vllm/gpt-6-astra \
  --lifecycle-timeout 7200
```

将模型名替换为你的 endpoint 实际提供的名称；需要时增加 `--effort`。
当前 CLI **没有 `--agent-config` 参数**。不使用 env 文件时，可通过宿主机的
`ASI_MODEL_*` 环境变量或 BenchFlow 原生模型路由配置运行。

- `--run-dir` 必须是不存在的新目录，不能位于 prepared/raw 内部。
- `--lifecycle-timeout` 包括 setup、构建、安装、连接和解题；默认 1800 秒。
  增大它不会覆盖 prepared 中的任务执行超时，两者先到者生效。
- solve 只解题、收集输出和清理，不执行 verifier；成功后的 `evaluation_status: pending` 正常。

正常摘要：

```json
{
  "status": "completed",
  "attempt_status": "completed",
  "collection_status": "collected",
  "prediction_status": "present",
  "cleanup_status": "completed",
  "evaluation_status": "pending"
}
```

### 3.6 独立评分

```bash
export RESULT_FILE="$RUN_DIR/$TASK_ID/${INSTANCE_ID}__${LEVEL}.json"

uv run python -m benchmarks.asi_bench.tests.smoke_score \
  --docker \
  --prepared "$PREPARED_DIR" \
  --result "$RESULT_FILE" \
  --output "$SCORE_DIR" \
  --timeout 600
```

这是当前可用的评分 CLI，内部调用生产函数 `scorer.score_result`；虽然模块名为
`tests.smoke_score`，执行的是真实评分，不是 dummy 评分。

- `--result` 传 ASI 风格结果 JSON，不是 `.outputs/` 或 BenchFlow 原生 `result.json`。
- `--output` 必须不存在；重复评分请换目录，不覆盖已有报告。
- 每次从已校验的 `prepared/environment/` 临时副本执行 Docker build，可使用构建缓存。
  **不要求原解题镜像仍存在，也不要求两个镜像 ID 相同。**
- `--timeout` 分别用于镜像构建和评分执行，不是整个流程的统一总时限。
- 评分通过临时只读 `/prediction/` 提供 Agent 输出及可信 instance `data/`；reference 分开挂载。
  不需要 Agent 提交原始输入数据，也不会修改原始解题输出。
- 评分结果独立保存，不回写 produce-only 结果的 pending 状态。

## 4. 目录与产物

```text
$ACCEPT_ROOT/
├── raw/
│   ├── sources.json
│   ├── instance/<task>__seed31415/       # prompt、data、reference 等
│   └── task_bundle/tasks/<domain>/<name>/
├── prepared/<task>/<id>/
│   ├── task.md
│   ├── task_manifest.json
│   ├── environment/
│   │   ├── Dockerfile
│   │   ├── runtime-profile.json
│   │   ├── runtime-constraints.txt
│   │   └── inputs/                      # prompt.md、task_info.json、任务输入
│   └── verifier/
│       ├── test.sh
│       ├── score_entry.py
│       ├── scoring.py
│       └── instance_parameters.json （可以删除）
├── run/
│   ├── run_metadata.json
│   ├── <task>/
│   │   ├── <instance>__<level>.json
│   │   └── <instance>__<level>.outputs/  # task 声明的 Agent 输出
│   └── benchflow/                       # 原生 rollout、日志、轨迹与运行目录
└── score/
    ├── asibench_score.json
    ├── reward.txt
    └── execution.json
```

评分前应保留整个 run 目录：评分还会校验 `run_metadata.json` 和结果引用的原生
rollout `result.json`，不只是读取 `.outputs/`。
`install-stdout.txt` 等是 BenchFlow 安装/执行日志；原生 `artifacts/` 可以为空，
它不是 ASI `.outputs/` 的替代目录。输出文件名由 task 契约决定，不是所有任务都需要 `certificate.json`。

### 4.1 `example/`：真实链路验证产物

适配器目录下的 `benchmarks/asi_bench/example/` 保存真实任务的**验证产物快照**，
用于查看下载、转换、Agent 解题和独立评分的输入输出，不是适配器运行时依赖，也不是参考答案模板。

```text
benchmarks/asi_bench/example/
├── raw/        # 下载的 seed31415 instance、reference、task bundle 和来源清单
├── prepared/   # 转换后的任务、Agent 环境及 verifier 文件
├── runs/       # 真实 Agent 解题结果、收集产物及 BenchFlow 执行记录
└── scored/     # 独立评分报告、reward 和容器执行记录
```

- `math.mpsc_safety_filter`：完整链路的成功验证样本；
  `scored/math.mpsc-1/asibench_score.json` 记录约 **99.9654 / 100**，两个 hard gates
  均通过，评分错误数为零。
- `math.levin_context_grid_search`：补充验证及问题排查样本；
  `scored/levin-b1-2/` 记录 **5 / 100**，存在下文说明的任务 runtime 异常，
  不应将其视为科学评分正常的成功样本或直接用于评价 Agent 能力。

这些是 non-official 本地验证记录，不代表所有任务或所有 level 均已验收。
快照中的 manifest/运行记录可能仍引用原验收机器的绝对路径，**不能假定复制目录后就能直接重评分**。
需要复现时，按第 3 节命令在新的独立目录中下载、转换、解题和评分；不要手工修改快照或摘要来绕过校验。

## 5. 如何判读失败和低分

**`status: completed` 不等于 Agent 解题正确，也不保证 task scorer 内部没有捕获异常。**

评分后检查完整报告，而不仅是终端的分数摘要：

```bash
uv run python - "$SCORE_DIR/asibench_score.json" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
for key in ("evaluation_status", "score", "max_score", "error", "gate_results", "score_details"):
    print(key + ":")
    print(json.dumps(report.get(key), ensure_ascii=False, indent=2))
PY
```

| 现象 | 检查方向 |
| --- | --- |
| `ModuleNotFoundError` | 使用 `uv run`，确认当前目录和依赖，而不是系统 Python |
| `FileExistsError` | 下载/run/score 要求新目录；不要传已存在的父目录 |
| `inventory mismatch` | 看 missing/extra 与摘要错误；未声明的普通 `.DS_Store` 已允许，不要改 manifest 绕过其他差异 |
| `execution_failed` / `prediction_status: missing` | 检查原始 traceback、模型路由、Agent 日志和预期输出契约 |
| 模型提示缺少 `OPENAI_API_KEY` | 检查模型路由；自定义 endpoint 示例使用 `vllm/<model>` |
| scorer 未注册或依赖导入失败 | 检查 Judge 限制、task 自定义 scorer 和镜像依赖 |
| `completed` 但分数低 | 看 `score_details` 的 message、details.errors、子项分数，区分任务表现与评分运行时错误 |

异常 traceback 会直接输出到 stderr，不脱敏、不截断。分享日志前检查是否含凭据或敏感请求信息。

## 6. 开发与回归测试

只有开发时的上游源码对照测试需要 `ASI_BENCH_SOURCE`（包含固定 revision 的 Git checkout）；
正常评分不读取此环境变量。更新 evaluator 的方法见下节。

```bash
ASI_BENCH_SOURCE=/path/to/ASI-Bench \
  uv run python -m pytest benchmarks/asi_bench/tests -q
```

下载回归覆盖固定清单、离线缓存复用、损坏缓存重取、下载失败拒绝与镜像构建前预取。
部分测试为显式开启的 Docker smoke，默认跳过；上述全套测试不是“所有真实任务均通过”的保证。
真实链路验收使用第 3 节命令，并检查完整评分报告。

维护边界：适配代码位于 `benchmarks/asi_bench/`，不修改 BenchFlow core 或 ASI 上游评分代码。
当前接受 task_meta/task_eval 拆分契约，也保留已有 `task.yaml` 读取兼容。
现有 prepared 对评分代码有身份绑定；仅兼容已知的 workspace 修复前 bridge v6，
评分执行使用新版 bridge 并记录实际身份，不允许任意历史 prepared 绕过检查。


## Evaluator 来源、维护与许可证

来源：`apexin-ai/ASI-Bench`，固定 revision
`5935b5f33549e348a8505bb355d3b6f4fe4a273c`。上游代码使用 Apache-2.0 许可证。
`evaluator_files.json` 精确列出原样使用的文件和摘要，以及本地替换方式。
初始化文件限制导入范围，`task_env.py` 替换原生环境管理实现；
不引入 generators、Agent adapters、orchestrator 或 LLM/VLM Judge。

升级时核对新 revision 的 7 个原始文件及 SHA-256，更新 `evaluator_files.json`，
审查依赖变化，并运行适配器测试。开发时可用
`materialize_evaluator(upstream_root, destination)` 从显式本地来源组装对照；
本地来源只需要清单中的 `copy` 文件，生成文件与 runtime 模板仍由适配器提供。
身份由清单、初始化内容和模板计算；不要手工修改已有 prepared 的摘要来绕过身份校验。

以下保留上游完整许可证文本（下载代码的来源与版权声明不作删改）：

<details>
<summary>Apache License 2.0 — ASI-Bench</summary>

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or Derivative
          Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2026 ApexIntelligence-AI

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

</details>
