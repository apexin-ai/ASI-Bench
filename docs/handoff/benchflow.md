# ASI-Bench 接入 BenchFlow：工作交接

交接日期：2026-10-14。面向接手开发、验证和上游贡献的同事。

**目前交接的是代码和技术结论。原个人服务器已下线，此前服务器、账号凭据、模型配置、Docker 镜像、缓存、运行目录和实验产物均按作废处理。** 本文不依赖旧服务器恢复，也不把旧路径当作可用资源。历史实验仅说明曾验证到哪里；接手后需要在团队环境重新构建并保存新证据。

## 1. 当前结论

BenchFlow 原生接入的 converter、verifier 和离线汇总已实现；完整评分一致性验收和上游收录尚未完成。

| 项目 | 交接状态 |
| --- | --- |
| 转换范围 | 固定源的 60 个正式 task 中，46 个候选、14 个排除；46 × B1–B4 = 184 个目录曾全部通过 structural 检查 |
| 历史真实对账 | 五个 task 的 OpenCode B1，加两个 task 的 Pi ACP B1，共 7 个非空、同产物双评分样本 |
| 覆盖程度 | 五题 × 四级 × 两 harness 的起步矩阵仅 7/40；更不是 46 个候选的全量覆盖 |
| 自动验收 | `parity_experiment.json` 仍为模板，规范 verify 应报告 `insufficient-evidence` |
| 自动对账脚本 | `parity_test.py` 的三个入口仍会显式报未实现；历史对账依靠私有操作脚本，并未形成完整自动化交付 |
| 上游贡献 | 截至最后工作记录，ASI 集成 PR 尚未创建；接手时重新核对上游 |

“结构检查通过”“模型能接通”“同一份产物评分相同”和“全量通过”是不同结论，不可互相替代。

## 2. 代码在哪里，先保全什么

以下是交接时本机代码位置，迁移后使用同事自己的目录。

| 仓库 / 分支 | 基线与用途 |
| --- | --- |
| `benchflow`，原位于 `/Users/erix/Documents/Apexin/benchflow` | 当前分支 `feat/asi-bench`，HEAD `7b152e617f76cbe21bb2e3af395886d550e6545e`；原生接入的主要代码在这里 |
| BenchFlow 个人 fork | 本机 origin 为 `git@github.com:Erix8/benchflow.git`；本地缓存的 `origin/feat/asi-bench` 与 HEAD 相同，未在本次交接中联网确认该远端 |
| BenchFlow 上游 | `https://github.com/benchflow-ai/benchflow`；最终贡献目标，不能假定上游已有 ASI 接入 |
| ASI-Bench | `https://github.com/apexin-ai/ASI-Bench`；提供公开 task bundle 和评分代码 |
| ASI 的旧 `benchflow-adapter-work` 分支 | `e72663e`，早期寄存方案；仅作历史参考，后续以 BenchFlow 仓库的原生 verifier 方案为准 |

**接手第一件事：将 BenchFlow 的完整 `feat/asi-bench` 分支及提交历史保全到团队仓库。** 不要依赖离职同事个人 fork 的长期可用性；若远端不可用，从现存本机 clone 转存代码。独立 Pi 兼容分支如仍需使用，也应一起转存。此文只整理交接，未执行仓库所有权迁移。

ASI-Bench main 中已有 `asibench benchflow-score`，但它是另一条 manifest bridge：BenchFlow 调度 `asibench run` 后按 manifest 评分。相关代码是 `ai4sci_bench/benchflow.py`，用法见 [现有指南](../guide/benchflow.md)。本次原生方案由 BenchFlow 自己运行 agent，并在其 verifier 中评分，不以 bridge 为执行入口，也没有完成 bridge 的弃用。

旧寄存分支曾带有不适合公开的示例资产；不要整体复制其 `example/`，也不要把它直接合入 ASI-Bench main。

## 3. 实现结构和关键契约

主目录为 BenchFlow 仓库的 `benchmarks/asi-bench/`：

| 文件 | 职责 |
| --- | --- |
| `benchflow.py`、`main.py` | 下载 / 读取固定源，转换 task × instance × level |
| `verifier_template/` | 沙箱内评分、产物收集和 reward 输出 |
| `evaluator_files.json` | 复制的评分框架文件、task helper、源 revision 和逐文件 SHA-256 |
| `benchmark.yaml` | 接入描述与 184 个转换目录的计数 |
| `parity_experiment.json` | 源 task 清单、排除原因、待完成覆盖矩阵 |
| `parity_test.py` | 待实现的自动对账入口，当前显式拒绝空证据通过 |
| `run_asi_bench.py` | 已实现 `summarize`；运行仍走 `bench eval run`，其独立 `run()` 入口不提供执行功能 |
| `README.md` | 当前适配器契约和限制 |

转换时固定两个不可变版本：

```text
ASI_REVISION = f13175a89dc9b4873f6306a3d31e46927c38f1a9
HF_REVISION  = 0fa14219cafdbab634d8b3cfbce238a8735a214f
HF dataset   = Apexintelligence-AI/ASI-Bench-seed31415
```

即使 ASI main 已更新，也不能直接替换这两个 pin。变更评分代码或源数据时，要同步 evaluator 清单与哈希、重新转换并重新对账。旧转换目录内是代码副本，更新模板不会自动更新已生成目录。

执行流程：HF 已物化实例及公开 task bundle → converter → BenchFlow task → 原生 agent → verifier → reward 与诊断。

- 只支持公开 seed31415，本地评分必须标为 non-official；seed42 不接入，不生成 GT。
- 14 个排除项为四个联网 task、九个 multimodal scorer task、一个 LLM Judge task；确切 ID 见 `parity_experiment.json`。转换器另有 submission sandbox、预建 task image 等排除规则，后续更新源时要重算清单。
- agent 输入在 `environment/inputs/data/`；评分用不可变的 `verifier/instance_data/data/`。另保留 `verifier/data/`，满足部分 scorer 读取 reference 相邻 data 的契约。
- reference 留在 verifier，BenchFlow 在 agent 结束后上传。当前采用 shared verifier，不能宣称具有 Harbor 式独立 verifier 容器的隔离保证。
- `evaluation.runtime: task` 以及直接导入依赖的 custom scorer，需要在 verifier 的隔离 Python 环境安装相应包；不能只给 agent 装依赖。
- 有效评分写 `reward.txt`、`reward.json` 和完整 `asi_score.json`；ASI 诊断字段放在 reward 的 `metadata` / `details`，不能任意平铺到顶层。
- 实际评分的声明产物归档到 `verifier/predictions/`，其 SHA-256 记录于 `reward.json.artifacts`，供同文件复核。
- 缺预测文件属于 submission failure，按 gate/scorer 合法评分；评估器故障写 `asi_error.json`、不写 reward、非零退出。低于满分的 BenchFlow `FAIL` 标签本身不表示评分链路失败。
- `summarize` 将合法零分纳入均值；评估器故障计入 `scorer_error_count` 并排除；未评分计入 `unscored_count`，全未评分时均分为 null。拒绝混合 ASI/HF revision。

## 4. 历史验证与不能继承的证据

2026-10-10 的原生 verifier 验证集为：

| Task ID | OpenCode B1 同产物对账 | Pi ACP B1 同产物对账 |
| --- | --- | --- |
| `math.levin_context_grid_search` | 已完成 | 已完成 |
| `math.mpsc_safety_filter` | 已完成 | 已完成 |
| `math.homotopy_poly_roots` | 已完成 | 未完成 |
| `robotics.particle_filter` | 已完成 | 未完成 |
| `robotics.rnea_inverse_dynamics_dh` | 已完成 | 未完成 |

这七个样本曾核对预测文件哈希、总分、hard/soft gate、scorer 的 score/max/pass 和稳定明细。部分早期 job 的 HF 元数据仍写 `main`；后来固定 revision、完成 184 个结构检查，并不等于已经在新 pin 下重跑了全部双评分。

Pi 后续实验受到上游 HTTP 524 / 500 影响；有的产生合法零分，有的被中断，均不能补作非空 parity 或干净的能力样本。此前暂停模型运行的决定不是本次交接自动恢复运行的授权。

旧 job、reference、预测、日志、私有 AWS 启动器和个人密钥不属于可继续使用的交接资源。新验收要重新获取公开数据、生成真实预测并归档，不要依据旧分数手填完整 coverage。

Pi ACP 历史兼容修复使用 Pi `0.81.1` 与 `pi-acp 0.0.34`。独立参考 PR 是 [BenchFlow #1158](https://github.com/benchflow-ai/benchflow/pull/1158) 和 [agents #77](https://github.com/benchflow-ai/agents/pull/77)，不是 ASI 集成 PR；链接只作历史索引，本次未核验其当前状态。不要与 ASI/Harbor 的 Pi `0.84.3` 配置混用。

## 5. 在新环境从零接手

建议使用团队管理的 Linux + Docker 环境，具备 GitHub、HF、Python/npm 依赖源和所选模型服务的访问条件。安装凭据由团队重新配置，密钥仅经私有配置或进程环境注入，不写命令参数、job config 或报告。

先转存分支、记录 HEAD，再按该仓库指南安装环境。以下在 BenchFlow 根目录执行；首次安装可能访问依赖源，但以下测试和结构检查不调用模型：

```bash
uv sync --extra dev --locked
uv run pytest -q tests/test_asi_bench_scaffold.py \
  tests/test_asi_bench_verifier.py tests/test_asi_bench_summarize.py \
  tests/test_asi_bench_coverage.py
uv run python benchmarks/asi-bench/main.py --help
```

从固定远端源转换单题 B1，使用新的输出目录（会下载公开数据）：

```bash
uv run python benchmarks/asi-bench/main.py \
  --task-ids robotics.particle_filter --levels B1 \
  --output-dir /tmp/asi-bf-new-tasks
```

离线转换需要事先准备完整的固定版本源：

```text
<source>/task_bundle/tasks/<domain>/<name>/...
<source>/instance/<task_id>__seed31415/{data/,reference/,instance_meta.json,prompt_b1.md,...}
```

设置 `ASI_BENCH_SOURCE` 指向该固定 ASI checkout 的根目录，用 `--source-dir <source>` 转换。该变量用于定位评分框架文件。CLI help 提到的 `--instance-dir` 目前并无对应参数，不要照此拼接命令。遇 GitHub API 403，可准备完整固定源后转为离线模式；下载重试修复不能消除服务端限额。

接着按 `bench eval run --help` 配置团队 endpoint、agent/model、Docker 和低并发，先运行单题。每个新 job 均保存 result、trajectory、verifier 诊断和预测哈希；复用同一份预测，用固定 ASI revision 的 `asibench score --repo seed31415` 独立复核。不可让模型重新生成一份输出再比较分数。

汇总和覆盖验收入口：

```bash
uv run python benchmarks/asi-bench/run_asi_bench.py summarize <jobs-dir>
uv run bench eval adopt asi-bench --verify
```

当前 verify 非零退出并报 `insufficient-evidence` 是预期。旧 `.venv/bin/bench` 曾因 shebang 指向已删除 worktree 而不可用，应在新环境重新安装。新环境实际 CLI 参数以安装版本的 help 为准。

## 6. 剩余工作与完成标准

1. **保全代码**：将个人 fork 分支迁入团队可维护的仓库，确认完整提交和 Pi 兼容改动可取到。
2. **重建基础链路**：跑定向测试、单题转换、空产物和评估器故障探针，再生成一份真实非空预测完成双评分。
3. **补自动对账**：实现 `parity_test.py`，包含相同文件集与 SHA-256、同 revision、有效状态、gate/scorer 稳定明细比较。私有服务器脚本不能作为唯一执行入口。
4. **扩大覆盖**：先补五题 × B1–B4 × OpenCode/Pi ACP 的起步矩阵；再完成当前 46 个候选 × 四级 × 两 harness 的覆盖，按现有清单为 368 个 cell。范围调整必须显式记录理由，不能只改 coverage 状态。
5. **收尾 PR**：完成证据后更新 `parity_experiment.json`，运行 verify、相关通用 verify/CLI 回归、Ruff 和格式检查，再向 BenchFlow 上游提交 ASI 集成 PR，按其人工 review 流程合入。

完成判据：新环境可复现、覆盖声明与逐 cell 证据一致、有效零分与评估器错误严格区分、verify 达到有证据支持的 `parity-confirmed`，且公开 PR 不含题面/reference/预测/私有模型日志。历史全文件格式检查存在漂移，需在提交前处理或明确记录。

## 7. 交接核对与来源

本次代码核对：BenchFlow HEAD `7b152e61`；四个适配器测试文件 **67 passed**。ASI 最新远端 main 已拉取到 `7651046`，其 manifest bridge 测试 **14 passed**。本次没有模型调用、Docker 实跑或全量回归，不能据此宣称完整集成验收。

本文件综合原 `.worktrees/` 中的 `benchflow-plan.md`、`benchflow-p5-handoff.md`、`benchflow-p4-handoff.md`，以及 `archive/benchflow-p3-2026-10-10/benchflow-p3-handoff.md`、`archive/benchflow-before-p3/` 下的 survey、report、shared-scoring 记录，并与当前代码交叉核对。历史目录被 Git 忽略，未作为同事使用本文的前置条件。

旧文档中的 44/60、Judge 12 题、Claude/Codex 起步方案以及“HF pin 待固定”等描述已被后续阶段取代；以本文的固定源清单和当前代码为准。后续维护人须同时负责 ASI/HF pin、evaluator 哈希、转换器及评分一致性测试。
