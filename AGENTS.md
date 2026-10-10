# ASI-Bench — 项目指南

> **重要：Claude 必须自主维护本文件。** 架构或约定变化时更新，保持简洁。

## Git 信息

- Remote: git@github.com:apexin-ai/ASI-Bench.git
- 默认分支: main

## ASI-Bench 公开边界

- 正式任务目录 `tasks/<domain>/<name>/` 公开 `task_meta.yaml`、只含
  评分/输出契约的 `task_eval.yaml`、可选 `custom_scorer.py`，以及
  `config/public_task_runtimes.json` 精确列出的 task runtime 文件；不跟踪
  benchmark prompts、`generation` 配置、GT 生成器、reference specs、参考
  答案或私有求解器资产。
- `config/public_scorers.json` 是正式任务公开评分器的精确 allowlist 和来源
  revision；正式任务严禁出现 `generate_gt.py`、`precompute_gt.py`、
  `reference_specs.md`、reference/ground-truth 目录或 `private_assets` / `reference_solver`。
- `config/public_task_runtimes.json` 是正式任务公开 runtime 文件的精确 task-level
  allowlist；其中的文件必须由 `task_meta.yaml` 显式声明并仅提供可复现执行环境，
  不得包含 prompt、reference、GT 或私有求解逻辑。
- 公开 scorer 只消费预生成的 instance/reference bundle，不得接受 seed
  或重建 GT。`config/public_scorers.json` 可精确列出通用 helper 和
  evaluator-only `*_eval_runtime.py`；这些 runtime 不得包含 generator、
  reference builder 或 hidden reference policy。custom scorer 必须显式区分
  submission failure 与 evaluator/runtime failure；后者设置
  `scorer_internal_error: true`，不得伪装成正常零分。
- `config/public_examples.json` 明确列出的五个公开示例任务是唯一例外，可保留
  B1–B4 prompts、GT 生成器、评分配置和 reference specs；不得扩展到其他任务。
- `tasks/_template/` 是框架级任务作者脚手架，不属于正式 benchmark 任务。
- seed31415 在 Hugging Face 公开 `reference/`，允许用 GitHub 的
  `task_eval.yaml` / `custom_scorer.py` 通过 `asibench score` 本地评分；
  本地报告必须标记 non-official 且不得覆盖 produce-only 结果。
- seed42 不公开 reference/GT；`task pull` 必须在下载和缓存复制两层
  过滤 `reference/`，`asibench score --repo seed42` 必须拒绝并引导 `submit`。
- `asibench submit` 只接受所有 instance ID 均属于 seed42 的结果；必须在打包、
  鉴权和联网前拒绝 seed31415、未知或混合 seed，且不得信任 `--benchmark-repo`
  绕过实例级校验。每个 seed42 结果还必须具有 `--sandbox os` 生成的、包含镜像
  identity 的 fail-closed Docker provenance，否则同样在打包和鉴权前拒绝。
- `asibench task submit --task-dir ...` 使用本地 PAT 将 Task 精确同步为 Portal Draft，
  然后打开 owner-only 页面供作者核对文件和字段；CLI 不执行最终 submit。首次登录由
  用户在 Portal Settings 手动创建/复制 PAT，CLI 隐藏输入、在线校验并以 0600 保存；
  CI 使用 `ASIBENCH_SUBMIT_TOKEN`，不得提供 token 命令行参数。
- Task 贡献的 `difficulty-check` 必须记录 B1–B4，但只以 B3/B4 平均分严格
  小于 40 为通过条件；B1/B2 不限分且报告为 `RECORDED`，CLI 不得允许把
  B3/B4 阈值调高到 40 以上，catalog flagged 也只检查 B3/B4。该检查必须在
  Docker `os` 沙箱运行，不设默认 agent；每个 `--agent` 必须对应显式含
  `model` 的 `--agent-config`，且至少包含一个多轮 harness。`direct_llm` 只能作为
  附加 baseline。CLI 必须在实例执行前显示全局进度条及当前 task、agent、
  B1–B4 level 和 instance，完成或失败后立即推进进度。Task Draft 的每组 `local_test_results` 必须记录 `sandbox: os`
  和 agent，且至少一组来自多轮 harness，并在联网前校验。
- `task_submission.yaml` 保存 Portal-only 作者证据，随 Revision 冻结供审核，但不得
  导出进 benchmark Task 仓库；正式任务目录仍不得公开该文件，只有 `_template` 可包含。
- `--instances-dir` 是只读输入；运行专属的 `framework_task_info.json` 必须写入
  output 目录，不得新增或覆盖 Hugging Face 拉取目录中的文件。
- produce-only 的数值零只是序列化占位符；报告不得将未评分结果显示为
  `0.0`，全未评分时应隐藏 per-task 分数表。seed31415 本地评分器内部错误
  必须记录 `evaluation_status: evaluation_invalid` 和 `final_score: null`，并从
  聚合分子、分母中排除，同时计入 `scorer_error_count`。评估器故障统一设置
  `scorer_internal_error: true`，并以 `failure_kind` 区分
  `evaluator_unavailable`、`evaluator_runtime_error` 和
  `missing_evaluator_input`；普通 submission failure 仍是有效零分。
- BenchFlow 适配只接受已物化的 seed31415 manifest：必须校验
  `seed: 31415`、`instance_id` 后缀、现有 `instance_dir/reference/`、prediction artifact
  目录和 task bundle。`benchflow-score` 不得接受 seed 生成请求、调用
  `generate_gt.py` 或评分 seed42。输出必须包含固定 schema 的 ScoreDetail、
  artifact SHA-256、scorer/task revision 和 harness/model/effort provenance。
- Harbor `harbor-verify` 只评分已物化的公开 seed31415 task/instance 与恢复的 agent
  产物，复用本地评分的隔离合并、task runtime 和错误分类。成功时仅写归一化
  `reward.json` 及完整 `score_detail.json`；评估器故障必须删除旧 reward、写诊断并
  非零退出。缺产物仍是有效零分；`--out` 与全部输入目录分离，seed42 拒绝。
- Harbor `harbor export` 当前只导出单个已物化的 seed31415 task/level；需要当前
  框架 wheel，输出含独立 verifier、最小 registry 和 wheel SHA-256。reference 与
  scorer 只进入 verifier 镜像，导出目录含公开 prompt/reference，不入本仓库。
  `evaluation.runtime: task` 使用专用 verifier 镜像，在构建时安装并检查 task
  依赖与 Python 约束，评分时复用镜像依赖且保持 fresh worker 隔离。
  verifier 基础依赖含 NumPy；agent/verifier 镜像标识绑定 task、instance，verifier
  另绑定 wheel SHA-256。
- BenchFlow 运行 `asibench run` 必须启用 `--fail-on-agent-error`，且不得只信任
  进程退出码；manifest schema v2 必须提供对应 run result JSON，并将
  `prediction_dir` 绑定到其 persisted outputs，分别报告 attempt 与 evaluation 状态。
- 原生 pi/opencode adapter 必须先解析 `api_key_env`/api_base_env 再做校验；provenance 只保留 endpoint 和环境变量名，禁止持久化 API key。CLI 版本必须用 `--version`/help 实测确认。
- pi/opencode 的 OS 镜像固定 pi `0.84.3`、opencode `1.17.15` 和 Node 22；
  prompt 通过 `docker run -i` 的 stdin 传入。`runtime.dockerfile` 只构建可复用的
  task 基础镜像，框架必须按当前 `agent_type` 单独叠加且只安装所选 CLI；无 agent
  时直接使用 task 基础镜像。agent image cache identity 必须绑定 task/base image、
  agent 类型与精确安装命令，produce-only result 也必须持久化原生 token cost。
- 持久化元数据的路径脱敏必须保留完整 HTTP(S) API endpoint，只替换 URL
  之外的宿主机绝对路径；agent 执行失败必须报告为 `attempt_status:
  execution_failed`，不得与 scorer 完成或低分混淆。sandbox 运行只在执行失败且
  返回完整的框架 timeout 标记时报告 `TIMEOUT`，不得解析 agent 日志中的自然语言。
- 模型调用记录必须与可见 trajectory 分离，使用版本化 sidecar 保存每次可观测
  completion 的边界、ID、生命周期、传输/进程结果、内容状态和重试关系；空、截断、
  不可用或无法解析的事件流必须显式记录 coverage gap。只有 attempted-call 分母及
  全部内容状态均可观测时才可报告数值 `empty_response_rate`，不得从非空消息推断为 0。
- `score`、`run-score`、`benchflow-score` 的 LLM/VLM Judge runtime override
  使用 `--judge-api-base`、`--judge-api-key-env`、`--judge-api-protocol`；key
  参数只接受环境变量名，不得把 secret 写入 CLI、`task_eval.yaml`、日志或报告。
  OpenAI-compatible endpoint 必须统一改写模型路由，文本/VLM 行为一致；显式兼容端点
  必须在请求体中保留任务配置的 temperature，不得套用 LiteLLM 原生 provider 的参数过滤；
  `run-score` 必须将 override 转发给每个评分子进程，并从 agent 子进程移除
  Judge selector 及专用的非标准 key 环境变量。
- `run-score` 串联 seed31415 的 `run` 与 `score`；支持 `--parallel` 和
  `--repetitions`，重复流程使用独立输出目录和进程；所有 repetition 的 task
  共用一个有界队列，前一轮尾部释放的槽位须立即由后一轮 task 补位。
  `score --parallel N` 先完整预检，再以至多 N 个 fresh spawn 进程并行独立 result，单个
  result 内的 gate/scorer/retry/num_judges 保持串行；协调进程按源顺序原子写报告。
  每个评分 job 必须在独立临时目录合并只读 instance `data/` 与持久化 outputs，拒绝
  symlink、路径逃逸及 outputs 覆盖 evaluator input，且不得修改源 `.outputs`。
  run 结果 JSON 明确记录 agent 未产出文件（无 `persisted_outputs` 且 code/data_files
  为空，或 `persisted_outputs.dir` 为 null）时按空 outputs 评分为 submission failure；
  其余缺失 `.outputs` 仍在预检中整批拒绝。
  `input.files.name` 可以是实例展开模板；评分只要求声明输入时存在安全的 `data/`
  根目录，并整棵复制其实际物化内容。预测文件缺失属于 submission failure，不能
  设置 `scorer_internal_error`。仅 `task_eval.yaml` 显式声明
  `evaluation.runtime: task` 的 scorer 会在 worker 启动前按
  `runtime.python` / `runtime.packages` 创建或复用隔离环境；普通 task 不得触发环境构建。
  环境构建优先使用 `uv`，缺失时只可用满足版本约束的当前 Python 回退到 `venv` / `pip`，
  scorer runtime 必须与 worker 的 Python major/minor 一致，失败必须标记
  `evaluator_unavailable`。
  `run-score` 在全部 agent task 完成后才逐 repetition 启动上述评分，整个流程不得
  出现 \(N \times N\) 嵌套并发。
- host-side Claude/Kimi harness home 必须同时按 benchmark execution 与
  instance run 隔离；execution 结束必须 teardown 清理。目录键须包含原始
  run key 哈希，Kimi 临时根初始化须支持并发。
- 科学 MCP 只通过显式 `--mcp-config` 注入 Claude/Codex 的隔离 home；默认
  restricted 运行不得继承 ambient MCP。配置必须严格校验，启用时固定为
  search mode；当前 `--sandbox os` 因宿主 GUI/许可证/二进制映射不明确而须
  fail-fast。catalog 只保存上游来源、前置条件和可编辑模板，不捆绑第三方软件。
- CAD MCP 上游兼容补丁仅存于 `scripts/mcp/cad-repairs/`，绑定源 revision 和
  SHA-256；只对显式指定的干净副本应用，不捆绑 CAD 软件、不自动更改本机凭据配置。
  patched stdio 冒烟通过不得升级为真实 CAD 业务认证；复测需临时 HOME 与端口保护。
- MCP E2E 脚本在 `scripts/mcp/e2e/`（布局与各 server 情况见该目录 README）：只 clone
  不 vendor 上游，`manifest.json` 绑定 40 位 revision 与 server 自己的 Python，依赖必须
  精确钉死（只允许 `==` 加 `exclude_newer`，或按平台提交带 SHA-256 的 `@EXPLICIT` lock
  无求解安装，或 Node server 用上游带 integrity 的 `package-lock.json` 做 `npm ci`），不安装
  系统包（`host_requirements` 只在 clone 前检查；`executables` 与 `npm ci` 都只用
  `setup.HOST_PATH`=`/usr/bin:/bin`，即 smoke server PATH 的尾部）。新增安装方式或
  主机检查只加一个 `INSTALLERS` / `HOST_PROBES` 注册项，manifest 条目只允许公共键加本
  方式字段。同一上游 repo 的多个 id 可用可选 `checkout` 键共用一份 checkout/.venv：组内
  repository/revision/python/install 及该安装方式的字段必须逐字相同，组名不得撞其他 id；
  `<id>.mcp.json` 与 MCP server 名仍按 id 一份（`mcp__<id>__<tool>` 不变）。
- smoke（`smoke.py <id>`）必须用 server 自身 venv 在进程外独立计算参考值（非 Python server
  的 venv 为空，参考值用闭式解或由 smoke 直接 subprocess 调宿主后端二进制产生，不经 server），并覆盖
  manifest 列出的全部工具：数值错误 FAIL、上游缺陷 WARN、已探测的已知数值缺陷用三态
  （正确 PASS、精确符合缺陷 WARN、其他 FAIL），联网 server 只用封闭历史窗口或不可变
  标识（如 NCBI protein GI、UniProt accession、固定发布的数据集文件）并关闭上游
  缓存；随上游版本漂移的量只记进报告，不得进 GT 或断言。共享流程与通用检查在
  `e2e_smoke/runner.py`，每个 server 只写
  `e2e_smoke/servers/<id>.py` 的参考值与专属检查。直连 smoke 通过不等于 agent E2E 通过。
- fake task 仅放 `examples/mcp-e2e-tasks/`（`status: test`，不进 `tasks/`），以
  `--params '{"seed":31415}'` 生成；参考值只在 `generate --sandbox task` 的隔离环境中
  计算，agent 以 `--sandbox none` 运行。scorer 只比对输出与 reference 且按产物独立给分
  （整体零分由 hard gate 负责）；「答案是否真的来自 MCP 工具」由
  `scripts/mcp/e2e/verify_run.py` 按 `e2e_check.json` 从 run 产物判定，框架评分契约不变，
  schema 以 `verify_run.py` docstring 为准。
- 只有后端求解器无法进入 task runtime 时（gpaw、quantum_espresso：求解器只存在于 server
  的 conda prefix）才允许 generate_gt 携带**实测表**：必须由入库的测量脚本对钉死的
  revision + lock 产出、整表替换不得手改，实测维度压到最小，其余一律由纯函数在 generate
  时推导，并有离线测试对账脚本记录的 server 自身答案；测量脚本须能 `--check` 与已提交表
  逐值对比，换 run host 先跑它。
- verifier 实现在 `e2e_verify/`（仅标准库，`verify_run.py` 只是 CLI）：spec 严格校验，
  未知键、非法枚举、悬空引用一律 `invalid_spec`；取值统一走 `Selector`、比较器共用，新
  值类型只加 extractor 或 comparator，跨 task 共用的比较器不得为了单个测试收紧；工具名
  按 `mcp__<server>__<tool>` 精确匹配，日志解析器按 `agent_name` 选择。
- 证据规则（最容易复犯）：持久化 stdout 会脱敏 user 事件，并把宿主路径换成占位符
  （`<home>`、`<workspace>`、`<run_output_dir>`、`<repo_root>`，其余 `<abs_path>`）。
  shell 命令原文、工具返回值和路径类工具入参都必须按同 call id 从 trajectory 补齐，
  含只是被脱敏的返回值与入参——否则返回或接收宿主路径的工具（`mol_to_sdf`、
  `export`/`import_cad_file`）永远无法校验。trajectory 只保留
  `core.trajectory.KEY_ARG_NAMES` 列出的入参，新增这类检查前先确认该键在列表里。
  判定必须先比较、只有「比不过且该值只有脱敏副本」才记 WARN coverage gap，不得判
  FAIL，也不得直接把脱敏值当不可观测（否则 scrub-tolerant 的比较会被降级）：证据缺口
  不是 agent 的错。链式入参的规范形（如 `output_file` 的文件名）在脱敏后仍完整时照常比较，
  不匹配即 FAIL。测试的 `persist_like_run` 必须调用真实
  `_sanitize_raw_artifact_text`，不得手写近似；测试不得依赖运行者的 `$HOME`
  （`tests/mcp_e2e/conftest.py` 已把它钉到 sentinel）。
- 客户端差异：Claude 对带 outputSchema 的 FastMCP 工具展示 `structuredContent`
  `{"result": ...}`，verifier 须先解包，extractor 遇到非对象 message 跳过而不抛错；
  trajectory 不保存图片内容，只在 tool_result metadata 记 `content_types` /
  `image_media_types`。Codex 证据是 `exec --json` 的 `mcp_tool_call` item，无 server/tool
  列表事件，`mcp_connected` 只能由必需工具的成功返回证明，其内置 `list_mcp_resources*`
  不算必需工具。
- `--mcp-config` 强制 search mode（Claude WebSearch/WebFetch、Codex web_search），
  `--sandbox none` 下 shell 也能联网；fake task 必须把这些工具纳入 bypass 检查。
- `tests/mcp_e2e/golden.json` 锁定每个 `verify_one` 的 verdict/各项检查/per-call 状态：
  纯重构必须逐字节不变，有意改变判定用 `MCP_E2E_GOLDEN=update` 整文件重生成并审 diff，
  `importorskip` 之后的测试不得喂 golden。离线测试全部在 `tests/mcp_e2e/`，共享构件只在
  `support.py`，通用约定在 `test_task_contract.py` 参数化，新 task 只写
  `test_task_<task>.py` 的数据与场景。
- 外部仿真 benchmark 接入通过 `ai4sci_bench.integrations`：ScienceAgentBench
  转换器只把源记录放入本地 `private/`，CFDLLMBench 使用本地固定 OpenFOAM
  镜像，SciAgentGym 工具逐实例 allowlist，COSMO-Agent 使用
  FreeCADCmd/FEM/Xvfb 镜像并显式声明 STEP、FCStd、VTK 输出。不得提交上游
  私有数据、商业软件或许可证。

## 任务生命周期

- Responses 翻译分支必须保留工具类型、原始 ID 和终止状态；未知字段须明确报错，
  不得修改全局 `litellm.drop_params`。旧 LiteLLM 不支持 custom-tool round trip 时
  明确拒绝并引导 native passthrough。该分支仍是 buffered SSE，不代表原生流式。
  Responses-only 参数及 reasoning/item-reference 历史在翻译分支须提前拒绝；
  不得将 LiteLLM 接收参数误认为下游 Chat 协议完整保留参数。
- Claude 对原生 Responses endpoint 仅在显式 `anthropic_via_responses: true` 时启用；
  必须请求 `reasoning.encrypted_content` 并以 execution 随机密钥签名、绑定会话历史的
  replay envelope 保留 reasoning ID/密文。篡改、跨 execution/会话 replay、缺失密文、
  未知参数/内容项和上游失败必须 fail closed；默认仍走兼容 Chat-only endpoint 的路径。

你收到任务后，按以下 9 步流程自主完成：

1. **领取任务** — 你已被分配任务，阅读本文件和项目代码理解上下文
2. **创建工作区**:
   - `git fetch origin`（如有 remote）
   - `git worktree add -b task-<简短描述> .claude-manager/worktrees/task-<简短描述> origin/main`
   - 进入 worktree 目录工作（后续所有操作在 worktree 中）
   - 如果 worktree 创建失败，直接在当前分支工作
3. **实现功能** — 编写代码，确保可运行
4. **提交代码** — `git add` + `git commit`，commit message 简洁描述改动
5. **Merge + 测试**:
   - `git fetch origin && git merge origin/main`（集成最新代码，如有 remote）
   - 运行测试（如有测试命令）
6. **自动合并到 main**（如有 remote）:
   - `git fetch origin main`
   - `git rebase origin/main`，如果冲突则自行 resolve
   - 如果成功：`git checkout main && git merge <task-branch> && git push origin main`
   - 如果这一步有任何失败，退回到步骤 5 重试
   - （纯本地项目跳过本步）
7. **标记完成** — 更新文档（必须在清理之前，防止进程被杀时状态丢失）
8. **清理** — 回到项目根目录:
   - `git worktree remove .claude-manager/worktrees/<worktree名>`
   - `git branch -D <task-branch>`
   - 如有 remote: `git push origin --delete <task-branch>`
9. **经验沉淀** — 在 PROGRESS.md 记录经验教训（可选）

### 冲突处理

rebase 发生冲突时：
1. 查看冲突文件: `git diff --name-only --diff-filter=U`
2. 逐个解决冲突
3. `git add <resolved-files> && git rebase --continue`
4. 如果无法解决: `git rebase --abort`，退回步骤 5

### 状态判断

- 通过 `git remote -v` 判断是否有 remote
- 有 remote → 必须完成步骤 6（merge + push）
- 无 remote → 跳过步骤 5 的 fetch、步骤 6 和步骤 8 的远程分支删除

## 文件维护规则

> **以下文件都由 Claude Code 自主维护，每次功能变更后必须同步更新。**

- **CLAUDE.md**（本文件）：架构、约定、关键路径变化时更新，只改变化的部分，保持简洁
- **README.md**：面向用户的文档，功能、使用流程变化时同步更新，保持与实际代码一致
- **TEST.md**：测试指南，新增功能时同步添加测试用例和文档
- **PROGRESS.md**：见下方「经验教训沉淀」

## 测试规范

**开发时必须主动使用测试，不是事后补充！**

- **改代码前**：先跑测试，确认基线全绿
- **改代码后**：再跑一遍确认无回归
- **新增功能**：同步新增测试用例，更新 TEST.md
- **修 bug**：先写复现 bug 的测试（红），修复后确认变绿

### 持续集成

- `.github/workflows/ci.yml` 在 push 和 pull request 上使用 `uv.lock` 运行
  Python 3.11/3.13 测试，并构建、检查和干净安装 wheel/sdist
- GitHub 分支规则应将稳定聚合检查 `CI required` 设为必需状态检查
- `.github/workflows/publish.yml` 只在 GitHub Release 发布时运行；标签版本必须与
  `pyproject.toml` 一致，并使用 `PYPI_API_TOKEN` Actions Secret 发布到 PyPI
- `uv.lock` 固定开发和 CI 环境；PyPI wheel 继续使用 `pyproject.toml` 的
  兼容依赖范围，不把库依赖钉死到 lockfile 版本

## 经验教训沉淀

每次遇到问题或完成重要改动后，要在 PROGRESS.md 中记录：
- 遇到了什么问题
- 如何解决的
- 以后如何避免
- **必须附上 git commit ID**

**同样的问题不要犯两次！**

## 注意事项

- 在 worktree 中工作时，不要切换到其他分支
- 完成任务后确保代码可运行、测试通过
