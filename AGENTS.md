# ASI-Bench — 项目指南

> **重要：Claude 必须自主维护本文件。** 架构或约定变化时更新，保持简洁。

## Git 信息

- Remote: git@github.com:apexin-ai/ASI-Bench.git
- 默认分支: main

## ASI-Bench 公开边界

- 正式任务目录 `tasks/<domain>/<name>/` 公开 `task_meta.yaml`、只含
  评分/输出契约的 `task_eval.yaml` 以及可选 `custom_scorer.py`；不跟踪
  benchmark prompts、`generation` 配置、GT 生成器、reference specs、参考
  答案或私有求解器资产。
- `config/public_scorers.json` 是正式任务公开评分器的精确 allowlist 和来源
  revision；正式任务严禁出现 `generate_gt.py`、`precompute_gt.py`、
  `reference_specs.md`、reference/ground-truth 目录或 `private_assets` / `reference_solver`。
- 公开 scorer 只消费预生成的 instance/reference bundle，不得接受 seed
  或重建 GT。`config/public_scorers.json` 可精确列出通用 helper 和
  evaluator-only `*_eval_runtime.py`；这些 runtime 不得包含 generator、
  reference builder 或 hidden reference policy。
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
  绕过实例级校验。
- `asibench task submit --task-dir ...` 使用本地 PAT 将 Task 精确同步为 Portal Draft，
  然后打开 owner-only 页面供作者核对文件和字段；CLI 不执行最终 submit。首次登录由
  用户在 Portal Settings 手动创建/复制 PAT，CLI 隐藏输入、在线校验并以 0600 保存；
  CI 使用 `ASIBENCH_SUBMIT_TOKEN`，不得提供 token 命令行参数。
- Task 贡献的 `difficulty-check` 必须记录 B1–B4，但只以 B3/B4 平均分严格
  小于 40 为通过条件；B1/B2 不限分且报告为 `RECORDED`，CLI 不得允许把
  B3/B4 阈值调高到 40 以上，catalog flagged 也只检查 B3/B4。
- `task_submission.yaml` 保存 Portal-only 作者证据，随 Revision 冻结供审核，但不得
  导出进 benchmark Task 仓库；正式任务目录仍不得公开该文件，只有 `_template` 可包含。
- `--instances-dir` 是只读输入；运行专属的 `framework_task_info.json` 必须写入
  output 目录，不得新增或覆盖 Hugging Face 拉取目录中的文件。
- produce-only 的数值零只是序列化占位符；报告不得将未评分结果显示为
  `0.0`，全未评分时应隐藏 per-task 分数表。
- BenchFlow 适配只接受已物化的 seed31415 manifest：必须校验
  `seed: 31415`、`instance_id` 后缀、现有 `instance_dir/reference/`、prediction artifact
  目录和 task bundle。`benchflow-score` 不得接受 seed 生成请求、调用
  `generate_gt.py` 或评分 seed42。输出必须包含固定 schema 的 ScoreDetail、
  artifact SHA-256、scorer/task revision 和 harness/model/effort provenance。
- BenchFlow 运行 `asibench run` 必须启用 `--fail-on-agent-error`，且不得只信任
  进程退出码；manifest schema v2 必须提供对应 run result JSON，并将
  `prediction_dir` 绑定到其 persisted outputs，分别报告 attempt 与 evaluation 状态。
- 持久化元数据的路径脱敏必须保留完整 HTTP(S) API endpoint，只替换 URL
  之外的宿主机绝对路径；agent 执行失败必须报告为 `attempt_status:
  execution_failed`，不得与 scorer 完成或低分混淆。

## 任务生命周期

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
