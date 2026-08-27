# 开发计划：pi / opencode 原生 Agent Adapter

> 状态：草案 · 负责人：待定 · 创建：2026-02
>
> 目标：为 [pi](https://github.com/earendil-works/pi-mono) 和
> [opencode](https://opencode.ai) 两个 CLI coding agent 提供 ASI-Bench
> 一等原生支持（`--agent pi_cli` / `--agent opencode_cli`），补齐
> `--agent-cmd` 通用文件交换模式缺失的沙箱、鉴权、轨迹与成本能力。

---

## 1. 背景

### 1.1 现状

内置 adapter 已有 11 个（`ai4sci_bench/adapters/`）：`direct_llm`、
`claude_code_cli`、`codex_cli`、`kimi_code_cli`、`mimo_code_cli`、
`antigravity_cli`、`codewhale`、`openhands`、`hermes`、`http_agent`、
`docker_agent`。所有 CLI adapter 共享 `SubprocessAgentAdapter` 基类。

新 harness 目前只能走 `--agent-cmd`（`CLIAgentAdapter`，文件交换模式），
与原生 adapter 的能力差距：

| 能力 | `--agent-cmd` | 原生 adapter |
|---|---|---|
| 沙箱 | `none` / `linux_ns` | + `task` / `os`（Docker） |
| 鉴权 | 用户自行解决 | api_key/api_base/api_protocol 注入、凭证挂载、端点识别 |
| 模型代理 | 无 | LiteLLM / TokenRouter / AnthropicRewrite 跨协议翻译 |
| 轨迹 | 单步 generic（除非自写 `_trajectory.jsonl` 且为 claude/codex schema） | 原生 JSONL → turns/tool_use/file_versions |
| Token 成本 | 无 | `CostInfo` 提取 |
| 假成功检测 | 仅 exit code | exit-0-but-API-error、中断 tool use 识别 |
| 工具隔离 | 无 | `ToolMode`（RESTRICTED/SEARCH/UNRESTRICTED） |

### 1.2 为什么要原生支持

- 轨迹分析是 ASI-Bench 论文的重要组成部分（2,600+ turns / 2,400+ steps）；
  通用模式下 pi/opencode 只能产出单步轨迹。
- leaderboard 上的 "agent–model configuration" 需要可复现的鉴权、effort、
  工具隔离配置；这些只有原生 adapter 能统一管理。
- Docker `os` 沙箱是跨平台可复现执行的标准路径，`--agent-cmd` 不支持。

---

## 2. 范围与里程碑

| 里程碑 | 内容 | 预估 | 交付物 |
|---|---|---|---|
| **M1** | `pi_cli` adapter（`none`/`linux_ns` 沙箱） | 2–3 天 | adapter + 单测 + 文档 |
| **M2** | pi 轨迹与成本管线 | 1–2 天 | `pi_extractor.py` + schema 检测 + 单测 |
| **M3** | `opencode_cli` adapter（`none`/`linux_ns` 沙箱） | 2–3 天 | adapter + 单测 + 文档 |
| **M4** | 两个 agent 的 Docker `os` 支持 | 1–2 天 | auth 挂载 + 镜像安装命令 + 冒烟 |
| **M5** | 端到端验证（seed31415 本地评分链路） | 1 天 | 冒烟报告、（可选）非官方分数 |

各里程碑可独立合并、独立发布；M2 依赖 M1，M4 依赖 M1/M3。

---

## 3. 通用实现蓝图（适用于任何新 CLI adapter）

### 3.1 Adapter 类骨架

新建 `ai4sci_bench/adapters/<name>_cli.py`，继承 `SubprocessAgentAdapter`，
参照 `kimi_code_cli.py`（最干净的近期范例）。必须实现的 hook：

| Hook | 职责 | 参考实现 |
|---|---|---|
| `__init__` | 参数校验（effort 白名单、api_protocol 白名单）、`supported_sandbox_modes` | `codex_cli.py:70` |
| `_build_command()` | headless 命令构建；prompt **不进 argv** | `claude_code_cli.py:_build_command` |
| `_get_stdin_input()` | 从 `workspace/prompt.md` 读 prompt 经 stdin 传入（规避 Windows ~8KB argv 上限，#27 教训）；通过代理时前置 agentic 指令 | `claude_code_cli.py:_get_stdin_input` |
| `_build_run_env()` | 合并 task env + 鉴权 env | `claude_code_cli.py:_build_run_env` |
| `_parse_log()` / `_raw_stdout_format()` | 返回 `"jsonl"` + 人类可读分 turn 摘要 | `claude_code_cli.py:_parse_claude_log` |
| `solve()` 后处理 | 成本提取 + 假成功（terminal error）检测 | `claude_code_cli.py:solve` |
| `_extract_usage_from_jsonl()` | `CostInfo(input_tokens, output_tokens, total_tokens)` | `claude_code_cli.py` 静态方法 |
| `setup()` / `teardown()` | 沙箱/代理生命周期 | `claude_code_cli.py` |

鉴权三模式约定（与现有 adapter 保持一致）：

1. **Local login**（默认）：无 `api_key`/`api_base`，用 harness 自身登录态。
2. **官方 API key**：仅 `api_key`，注入对应环境变量。
3. **第三方端点**：`api_base` + `api_key` + `api_protocol`（`openai`/`anthropic`），
   优先 harness 原生 provider 配置（如 Codex 的 `CODEX_HOME` 模式），
   协议不兼容时回落 LiteLLM 代理。

### 3.2 注册点清单（7 处，缺一不可）

| # | 文件 | 改动 |
|---|---|---|
| 1 | `ai4sci_bench/adapters/__init__.py` | import + `__all__` |
| 2 | `ai4sci_bench/cli.py` `_build_agent_metadata()` (~L175) | `adapter_class` 分支 |
| 3 | `ai4sci_bench/cli.py` `_build_agent()` (~L2030) | 构造分支 |
| 4 | `ai4sci_bench/cli.py` `_AGENT_CLI_BINARY` (~L101) | 二进制名 → banner 安装检测 |
| 5 | `ai4sci_bench/cli.py` `--agent` option help (~L968) | 名称列表 |
| 6 | `ai4sci_bench/runner/task_image.py` `AGENT_INSTALL_COMMANDS` | Docker 镜像安装命令（M4） |
| 7 | `ai4sci_bench/runner/os_sandbox.py` `_prepare_auth_mounts()` + `needs_api_network` 白名单 (~L277/L579) | Docker auth 挂载 + 联网放行（M4） |

可选第 8 处：`ai4sci_bench/runner/orchestrator.py`
`_detect_jsonl_trajectory_schema()` (~L1184) 加 schema 识别（M2/M3）。

---

## 4. M1 + M2：pi_cli 设计

### 4.1 pi CLI 事实清单（已核实 v0.x，随版本更新需复核）

- 非交互：`pi -p "prompt"`（print 模式）；`pi --mode json`（JSONL 事件流到 stdout）；
  `pi --mode rpc`（stdin/stdout RPC）。
- prompt 可经 stdin 管道：`cat prompt.md | pi -p`。
- 免审批 headless：`--no-approve` / `-na`（或 `--approve`）。
- 工具限制：`--tools read,grep,find,ls`（ToolMode 映射基础）。
- 会话：JSONL 树状结构，存于 `~/.pi/agent/sessions/`（按工作目录组织）；
  `--no-session` 为 ephemeral 模式。
- 模型选择：`--model`（provider/model 语法）；认证走 `/login` 订阅或 API key。
- 安装：`npm install -g @earendil-works/pi-coding-agent`（以官方 README 为准）。

### 4.2 命令构建

```
pi --mode json --no-approve [--model <model>] [--tools <map(tool_mode)>] [- <prompt via stdin>]
```

- `_raw_stdout_format()` → `"jsonl"`；`--mode json` 的事件流即原生轨迹。
- 鉴权：pi 复用标准 provider 环境变量（`ANTHROPIC_API_KEY` 等）；
  `api_base` 场景需生成临时 provider 配置或经 LiteLLM 代理——
  **实现前先核对 pi 的 custom-provider 文档**（`docs/custom-provider.md`）。

### 4.3 ToolMode 映射

| ToolMode | pi 参数 |
|---|---|
| RESTRICTED（默认） | 核心 Bash/Read/Write/Edit/Glob/Grep 等编码工具集 |
| SEARCH | RESTRICTED + WebSearch/WebFetch（以 pi 实际工具名为准） |
| UNRESTRICTED | 不传 `--tools` |

### 4.4 轨迹提取（M2）

- `pi --mode json` 事件类型：`agent_start/agent_end`、`turn_start/turn_end`、
  `message_start/update/end`、`tool_execution_start/end`（含 `toolName`、
  `args`、`isError`、`usage`）。
- 新增 `ai4sci_bench/trajectory/pi_extractor.py`：
  `extract_from_jsonl(raw, instance_id) -> Trajectory`，映射
  turn/message/tool 事件 → `TrajectoryStep`；从 `usage` 提取 `CostInfo`。
- `orchestrator._detect_jsonl_trajectory_schema()` 加规则：
  事件含 `agent_start` / `tool_execution_start` → `"pi"`。
- `file_history.extract_file_versions()` 评估是否可从 pi 事件流重建
  文件版本；不可则记录为已知限制。

### 4.5 Docker `os` 支持（M4）

- `AGENT_INSTALL_COMMANDS["pi"]` = npm 全局安装命令。
- auth 挂载：`~/.pi/agent/` 只读挂载进容器（实现时核对凭证文件精确路径，
  避免挂载整个目录引入不必要的宿主信息——参照 `~/.codex/auth.json` 的
  单文件挂载模式）。
- `needs_api_network` 白名单加 `"pi"`、`"opencode"`。

---

## 5. M3：opencode_cli 设计

### 5.1 opencode CLI 事实清单（多项需在实现前复核）

- 非交互：`opencode run [message..]`；模型选择 `--model provider/model`。
- 认证：`opencode auth login`（凭证存 `~/.local/share/opencode/auth.json`）；
  provider API key 也接受标准环境变量。
- 配置：项目级 `opencode.json` / 全局 config，含 provider/model/permission 块；
  权限模式需设置为免审批以支持 headless 长任务。
- 安装：`npm install -g opencode-ai`（以官方文档为准）。
- **待复核（开放问题 O1）**：`opencode run` 是否有 JSONL 结构化 stdout
  或可导出会话 JSONL。若有 → 走 pi 同款轨迹管线；若只有会话文件 →
  adapter 在 solve 结束后定位最新 session 文件并转换为 `_trajectory.jsonl`。

### 5.2 命令构建

```
opencode run [--model <model>] [--config <tmp-config>] - <prompt via stdin 或 argv 文件>
```

- 第三方端点优先复用 opencode 原生 provider 配置（生成临时 config，
  参照 kimi 的 `KIMI_CODE_HOME` 临时目录模式），协议不兼容再回落 LiteLLM。
- ToolMode 映射待核对 opencode 的工具/权限配置项后填入。

### 5.3 轨迹与成本（依赖 O1 结论）

- 方案 A（首选）：原生 JSONL stdout → `opencode_extractor.py`。
- 方案 B：run 结束后解析 `~/.local/share/opencode/` 会话存储 →
  转换为 Trajectory；成本从消息 usage 字段提取。
- 两方案都需在 `_detect_jsonl_trajectory_schema()` 或 adapter 层挂接。

---

## 6. 测试计划

> 遵循 AGENTS.md 测试规范：改前跑基线、改后回归、新增功能同步新增测试。

### 6.1 单元测试（不打真 API，全部 mock）

新建 `tests/test_pi_cli_adapter.py`、`tests/test_opencode_cli_adapter.py`，
用例对照 `test_kimi_adapter.py` / `test_mimo_adapter.py` / `test_adapters.py`：

- 命令构建：model/effort/tool_mode → 正确 flag；prompt 不进 argv；
  `shlex.quote` 占位符替换安全（复用 `test_adapters.py::TestCLIAgentAdapter` 断言）。
- 鉴权解析：三模式组合；非法 `api_protocol` 报错；env secret 读取（含未设置报错）。
- stdin 输入：从 prompt.md 读取；代理模式前置 agentic 指令。
- 日志解析：构造样例 JSONL fixture（覆盖正常 turn、tool call、错误事件、
  usage 事件），断言人类可读摘要、`CostInfo`、terminal-error 检测
  （exit 0 + is_error / interrupted → `RunStatus.FAILED`）。
- 注册完整性：`_build_agent("pi_cli")` 返回正确类；`_AGENT_CLI_BINARY` 命中。
- 轨迹：pi/opencode JSONL fixture → extractor 输出 `Trajectory` 摘要与步骤数；
  schema 检测函数识别新事件类型。

### 6.2 回归

```bash
uv run pytest tests/ -x        # 改前基线全绿；改后再次全绿
```

### 6.3 端到端冒烟（需真实 key，手动）

```bash
# 单实例调试
asibench run --agent pi_cli \
  --agent-config '{"model": "<provider/model>"}' \
  --instances-dir hf_instances_seed31415/ \
  --prompt-levels b3 --params '{"<task>": "<debug instance>"}' \
  --sandbox linux_ns --output-dir out_smoke/

# 检查点：
# 1. workspace 内产出任务要求的 output files
# 2. result JSON：raw_stdout_format=jsonl、trajectory_summary 非空、cost 非空
# 3. 人为构造 API 错误 key，验证 FAILED + error_message（假成功检测）

# seed31415 本地评分全链路（结果标记 non-official）
asibench score --repo seed31415 --results-dir out_smoke/ \
  --instances-dir hf_instances_seed31415/ --tasks-dir ./
```

M4 追加：`--sandbox os` 冒烟，验证镜像构建、auth 挂载、容器内联网。

### 6.4 文档同步

- `README.md`：内置 agent 安装表加 pi / opencode 两行 + 示例命令。
- `docs/guide/getting-started.md`：`--agent` 列表更新。
- `TEST.md`：新增测试用例说明。
- `PROGRESS.md`：每个里程碑记录 commit ID。

---

## 7. 风险与开放问题

| 编号 | 风险/问题 | 缓解 |
|---|---|---|
| O1 | opencode 结构化输出能力未确认（5.1） | M3 第一步先做 CLI 勘察 spike；两套轨迹方案备选 |
| O2 | pi/opencode 迭代快，CLI flag 与事件 schema 可能漂移 | fixture 标注测试时版本；adapter 参数校验给出明确版本提示 |
| O3 | effort 语义不对齐（claude low–max vs pi/opencode 各自档位） | adapter 内维护映射表并在 provenance 记录原始值；文档说明 |
| O4 | harness 系统提示差异导致分数不可比 | 与现有 18 个配置同口径：默认 RESTRICTED 工具隔离；论文口径的对照实验由维护者决策 |
| O5 | Windows argv/编码问题（#27、GBK 教训） | prompt 一律走 stdin；测试含非 ASCII prompt 用例 |
| O6 | Docker auth 挂载泄漏宿主凭证 | 只挂最小凭证文件（单文件 > 整目录）；挂载只读 |

## 8. 验收标准

- [ ] `asibench run --agent pi_cli` / `--agent opencode_cli` 在 `none`、
      `linux_ns` 沙箱下完成 b1–b4 全 level 运行，产出合规 result JSON
- [ ] 轨迹 summary 步骤数 > 1（真实多步轨迹，非 generic 单步）
- [ ] `CostInfo` 从原生事件提取成功
- [ ] API 错误、超时、中断场景均映射到正确 `RunStatus`
- [ ] 新增单测全绿，`uv run pytest tests/` 无回归
- [ ] README / getting-started / TEST.md 同步更新
- [ ] （M4）`--sandbox os` Docker 冒烟通过
- [ ] （M5）seed31415 本地评分链路跑通，non-official 报告生成
