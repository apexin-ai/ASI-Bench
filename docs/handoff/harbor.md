# ASI-Bench 接入 Harbor：工作交接

交接日期：2026-10-14。面向接手开发、验证和发布的同事。

**目前仅交接代码与技术结论。个人服务器已下线，原服务器、私有配置、密钥、Docker 镜像、临时目录和实验产物均按作废处理。** 旧记录中的 AWS 资源占用、运行路径和“恢复后继续”步骤不再适用；请在团队新环境从代码重新构建和验收。

## 1. 当前结论

已实现单个公开 seed31415 task/level 的 Harbor 导出和独立 verifier 评分入口。五题 B1 的离线导出、空输出、旧产物回放和同文件评分曾通过；真实 Harbor agent 的完整闭环只覆盖 particle filter B1。全任务覆盖、可复现分发和统一汇总尚未交付。

| 项目 | 已完成 | 尚未完成 |
| --- | --- | --- |
| 导出器 | 单个 task × instance × B1–B4 level，最小 registry，独立 verifier | 批量转换、完整任务适用范围审计 |
| 评分入口 | 复用本地评分 staging、task runtime、错误分类 | 全任务 / 特殊 scorer 的真实容器验收 |
| 五题 B1 | 导出、nop、历史预测回放与独立评分对账 | 五题均由 Harbor 原生 agent 新生成非空预测 |
| 真实 agent | particle filter B1 的 Claude Code 完整 trial；Pi Chat 接通冒烟 | GPT 路由稳定性、其他题和 B2–B4 覆盖 |
| 分发与汇总 | 本地导出目录及 registry | 团队数据集发布、可复现镜像、ASI 同口径聚合 |

## 2. 代码位置与分支状态

| 仓库 | 交接基线 |
| --- | --- |
| ASI-Bench | `https://github.com/apexin-ai/ASI-Bench`，**功能在 `feat/harbor-verify` 分支**，HEAD `d7b712019338383f132eb3e26e4c438c344aa49c`，与本地缓存的同名 origin 分支一致 |
| ASI 最新 main | 本次 fetch 后为 `76510466cfb9837646c6ca5599f90015b663d502`，版本 0.2.0；**尚不包含 Harbor 两个实现模块及命令** |
| Harbor | 本机 origin 为 `git@github.com:harbor-framework/harbor.git`；原目录 `/Users/erix/Documents/Apexin/harbor`，HEAD `09f5b964f8e5075f7e52fd08c9994dfc80ad5d24` |
| 历史运行版本 | Harbor `0.24.0`；功能分支 ASI wheel 为 `0.1.6`。这是历史兼容基线，不代表推荐安装最新版本后无需复测 |

**不能只安装 ASI main 或 PyPI 0.2.0 就假定存在 `harbor export` / `harbor-verify`。** 先保全功能分支，接入当前主线时处理后续 task runtime 等变化，重新运行相关测试并构建 wheel。此次交接没有把 Harbor 功能合入 main。

首版代码都在 ASI-Bench，不需要修改 Harbor 仓库。后续若 Harbor 契约变化或要贡献官方 adapter，再单独评估上游改动。开始实际适配前应核对 Harbor 工作树、更新到团队选定版本并记录 SHA；旧 HEAD 仅用于复现基线。

关键提交：

| 提交 | 内容 |
| --- | --- |
| `545f7c1`、`c311c25` | `harbor-verify`、无效 Judge 配置前清理旧 reward |
| `acd2f46` | 单个公开 task 的 Harbor 导出 |
| `61bdfb4` | task runtime 专用 verifier、依赖 / Python 检查、镜像身份 |
| `d7b7120` | P1 离线验证经验记录 |

主要文件是 `ai4sci_bench/harbor_export.py`、`ai4sci_bench/harbor_verify.py` 和 `ai4sci_bench/cli.py`；测试为 `tests/test_harbor_export.py`、`tests/test_harbor_verify.py`，共用本地评分逻辑位于 `ai4sci_bench/local_scoring.py`。

## 3. 导出和评分如何工作

```text
公开 task bundle + 已物化 seed31415 instance + 本次源码构建的 wheel
  → asibench harbor export
  → Harbor 原生 agent 在 agent 容器解题
  → Harbor 收集 artifacts 并恢复到独立 verifier 容器
  → asibench harbor-verify
  → reward.json + score_detail.json
```

一次导出生成一个数据集目录：

```text
<export>/
  registry.json
  export_manifest.json
  tasks/<task-slug>-b1/
    task.toml
    instruction.md
    environment/{Dockerfile,workspace/data/...}
    tests/{Dockerfile,test.sh,task_bundle/,instance/,wheels/}
```

- 只接受匹配的公开 seed31415 实例、非空 reference、指定 level 的 prompt 和真实 wheel；seed42 拒绝，不生成 GT。
- 输入实例只读；输出必须与 task、instance、wheel 分离，导出目标目录必须不存在。每次导出使用独立目录。
- agent 镜像仅放运行依赖和输入；reference、task scorer、wheel 只进入 verifier 构建上下文。导出目录含公开题面/reference，不提交到 ASI-Bench GitHub。
- `output.files` 映射成 `/workspace/...` artifacts；Harbor 按原路径恢复到 verifier，脚本读取 `/workspace` 的预测。
- `[verifier] environment_mode="separate"`；`tests/Dockerfile` 必须内置 `/tests/test.sh` 及全部评分文件，不能依赖评分时上传 tests。
- `reward.json` 仅写 `{"reward": final_score/max_score}`；原始分数、gate/scorer 明细、状态与 provenance 放 `score_detail.json`，避免额外数值被当作 reward 维度。
- 缺预测属于合法 submission failure，可返回零分；评估器缺依赖、缺 reference 或运行故障需清除旧 reward、写诊断、非零退出，标记 `evaluation_invalid` 和 `scorer_internal_error: true`。
- 错误类别沿用 `evaluator_unavailable`、`evaluator_runtime_error`、`missing_evaluator_input`。不能把 verifier 故障归因成 agent 零分。
- `evaluation.runtime: task` 使用专用 verifier 镜像预装 task 包并检查 Python 约束；运行时复用依赖，评分仍在 fresh worker 隔离执行。内部预装标记只适用于导出器构建的专用镜像，不应在通用宿主环境手工打开。
- 镜像 identity 绑定 task/instance，verifier 另绑定 wheel SHA-256。基线镜像固定 Python 3.12，不能因此宣称支持所有任务的 Python / Dockerfile / service 要求。

## 4. 历史验收到哪里

P0 曾用 Harbor Claude Code 在 particle filter B1 生成六个真实产物；与 `asibench score` 同文件对账为 `55.56304335672723/100`，Harbor reward 为 `0.5556304335672723`。后续专用 key 的独立 Claude Code trial 同样完成，得 `57.274249786058746/100`，同产物哈希、总分和稳定 gate/scorer 明细相同。两次是不同预测，分数无需相同，也不能据此声称各 harness 能力可比。

P1 的五题 B1 离线结果：

| Task ID | nop / 评估器故障探针 | 旧预测回放与独立评分 | Harbor 原生 agent 完整非空样本 |
| --- | --- | --- | --- |
| `math.mpsc_safety_filter` | 通过 | 通过 | 待补 |
| `math.levin_context_grid_search` | 通过 | 通过 | 待补 |
| `math.homotopy_poly_roots` | 通过 | 通过 | 待补 |
| `robotics.particle_filter` | 通过 | 通过 | Claude Code 已完成 |
| `robotics.rnea_inverse_dynamics_dh` | 通过 | 通过 | 待补 |

回放的是此前 BenchFlow agent 的预测文件，只证明转换、传递和评分，不代表新 Harbor agent 解题。历史 Pi `0.84.3` 的 Chat 冒烟经历三次 HTTP 524 后完成一次 bash 调用，最终因四轮上限未产出声明文件，合法零分；只能证明接通，不能计作求解通过或稳定性验证。

上述原始 job、预测和镜像现均不作可用交付。不能用历史摘要代替接手后的可复核产物；新验收需重新生成并保存证据。

## 5. 新环境重建步骤

先在团队仓库保全功能分支。在独立 checkout 使用该分支安装开发环境并构建 wheel；先不要和当前 main 的包混用：

```bash
# 在 feat/harbor-verify 的独立 ASI-Bench checkout 中
uv sync --extra full --group dev --locked
uv run pytest -q tests/test_harbor_verify.py tests/test_harbor_export.py
uv run asibench harbor export --help
uv run asibench harbor-verify --help
uv build --wheel
```

确认构建出的具体 wheel 文件名，记录源码 SHA 和 wheel SHA-256。重新拉取公开实例（需要 HF 访问）：

```bash
uv run asibench task pull --repo seed31415 \
  --tasks robotics.particle_filter --output-dir /tmp/asi-harbor-new-instances

uv run asibench harbor export \
  --task-dir tasks/robotics/particle_filter \
  --instance-dir /tmp/asi-harbor-new-instances/robotics.particle_filter__seed31415 \
  --level b1 --wheel dist/<本次构建的具体wheel文件名> \
  --output-dir /tmp/asi-harbor-new-export
```

上面的 wheel 占位符须替换；不要选旧 wheel 或用会匹配多个版本的通配符。对需要严格复现的数据，另记录 HF dataset commit，按该 commit 获取快照；不能把默认 pull 当作固定版本证据。task bundle、wheel 和独立评分源码也要保持匹配。

在单独的 Harbor 环境安装团队选定版本，核对 `harbor --version`、`harbor run --help`，先做无需模型的空输出探针：

```bash
harbor run -p /tmp/asi-harbor-new-export/tasks/robotics-particle-filter-b1 -a nop
```

预期 verifier 正常完成，缺产物由 gate/scorer 得到有效零分。随后只在导出副本中测试缺 reference 与旧 reward 清除：要求非零退出、诊断为评估器错误、无遗留 reward，不能损坏原实例。

团队模型账号、endpoint 和预算准备完成后，选择 Harbor 原生 agent 做一次新的真实运行：

```bash
harbor run -p /tmp/asi-harbor-new-export/tasks/robotics-particle-filter-b1 \
  -a claude-code -m <团队可用的模型名> -n 1
```

认证通过团队私有配置注入进程环境，避免把密钥写入 `--agent-env` 等会持久化的参数或 job config。旧专用 key 和旧网关均不作为前置条件。具体参数、认证变量和 artifact 布局以选定 Harbor 版本为准。

从新 trial 取实际持久化产物，校验逐文件 SHA-256，再用相同 task bundle / instance / wheel 对应版本运行 `asibench score --repo seed31415`。本地 score 需要 ASI run-result JSON；可参考 `tests/test_harbor_verify.py` 的 fixture / 对账方式生成与真实预测绑定的记录，不能拿 Harbor result JSON 直接冒充 ASI result。当前没有交付一条完整自动执行此步骤的通用命令。

如需直接复核恢复后的 artifact 目录，可先调用以下评分入口；这只验证 verifier，不能替代与 `asibench score` 的独立对账：

```bash
uv run asibench harbor-verify \
  --task-dir tasks/robotics/particle_filter \
  --instance-dir /tmp/asi-harbor-new-instances/robotics.particle_filter__seed31415 \
  --outputs-dir <恢复后包含analysis.py和results的目录> \
  --prompt-level b1 --out /tmp/asi-harbor-new-verify
```

保存 result、trajectory、产物哈希、score detail、独立评分报告、task/instance 来源 revision、wheel 和镜像 identity；验收应同时比较总分、max、hard/soft gate、evaluation 状态和稳定 scorer 明细，排除独立计时等非确定字段。

## 6. 已知问题与后续工作

1. **功能分支集成**：Harbor 功能尚未进入 main；先与当前主线合并、处理冲突和回归，再决定发布版本，不能直接使用 main 构建的 wheel。
2. **汇总语义不同**：历史 Harbor 默认 mean 把缺失 reward 当零，而 ASI 排除 `evaluation_invalid`。必须提供并验证 ASI 口径的汇总或明确标注 Harbor 原生均值。HF Git `--repo -d` 不会自动加载同目录 `metric.py`；Harbor Hub package 的自定义 metric 是待验证方案。
3. **网络策略待收敛**：历史 Docker provider 拒绝 `no-network`，当前导出配置使用 `public`。不能声称实现模型 endpoint 以外断网；换版本 / provider 后重新测策略，结合团队要求决定 allowlist。
4. **构建和镜像复用**：历史 particle replay 曾误复用另一题 verifier 脚本；通过 `--force-build --no-delete` 重建后恢复。新运行需检查镜像 task/instance 标签、wheel SHA 和镜像内 `test.sh`。旧代理、预下载依赖镜像全部重建。
5. **路由真实性与稳定性**：历史 GPT Responses 路由在缺顶层 `instructions` 时出现额外提示；Codex CLI 实际请求没有该字段，因此不能靠示例请求覆盖字段推断 Codex 已可用。新路由需分别验证 Chat、Messages、Responses 的工具往返和长任务稳定性，必要时由服务端核对实际送模输入。Chat / Messages 的客户端成功不能证明不存在隐藏提示。
6. **补足真实覆盖**：先完成五题 B1 新 agent 非空产物与同文件评分，再扩 B2–B4、更多 agent 和 task。Judge、需要嵌套 Docker、预建 task image / 服务 / 特殊 Python 依赖另行盘点，不引用早期“约 47 题”“60 × 4 已可发布”的估算。
7. **发布交付**：选择团队控制的 HF / registry 位置，批量 registry 生成、依赖锁定、基础镜像 digest、wheel 来源和 SHA、重建流程均需完善。逐次单题导出会各写一份最小 registry，不能直接覆盖到同一目标冒充批量数据集。

完成标准：团队环境可复现导出与独立评分；承诺范围内每个 cell 均有真实预测和同文件对账证据；评估器错误不计零；发布目录符合公开边界且有可复现构建与聚合流程。是否贡献 Harbor 官方目录需另行确认，不应公开私有参考解来满足其可能的收录要求。

## 7. 交接核对与来源

本次在 `feat/harbor-verify` 基线上执行两个专属测试文件：**20 passed**。没有 Docker、模型调用或全量回归。历史 P1 全量运行并非全绿，曾因受限网络安装失败等出现 26 failed、5 errors，其余失败未全部归因；不得在交接中改写成全量通过。

本文综合原 `.worktrees/harbor-p1-handoff.md`、`harbor-design.md`、`harbor-report.md`、`archive/harbor-p0-2026-10-10/harbor-p0-handoff.md`、`harbor-gateway-evidence-20261010/README.md`，并与功能分支代码核对。上述被忽略的本机记录只作历史来源，接手不要求访问原服务器或临时目录。旧阶段文档中的共享资源优先级、默认凭据、剩余磁盘和网关恢复步骤均已失效。
