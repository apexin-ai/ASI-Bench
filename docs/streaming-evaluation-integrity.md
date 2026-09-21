# Claude Code 评测：流式、空回复、转换器与超时

本文说明 ASI-Bench 已实现的修复、启用方法和验证边界。Qwen自部署以及走普通
OpenAI-compatible协议的GLM/Gemini/GPT共享转换代理；原生Anthropic接入使用独立保护层。
`claude -p`主对话默认请求流式；`--output-format stream-json`控制CLI日志格式，
不能保证代理也向模型发流式。以下结论不涉及商业网关内部如何调用供应商。

## 1. ASI转换器改了什么

```text
CC /v1/messages → ASI LiteLLM bridge → 模型 /v1/chat/completions
                   OpenAI响应转换为Anthropic事件
```

旧代码收到`reasoning`或`reasoning_content`时创建index=0的thinking块。
随后收到`content`时只检查index是否存在，没有检查类型，直接把text_delta写进
thinking块。CC可能因此放弃原响应，重新请求stream=false。

```text
旧版：start thinking(0) → thinking_delta(0) → text_delta(0)  [类型错误]
修复：start thinking(0) → thinking_delta(0) → stop(0)
      → start text(1) → text_delta(1) → stop(1)
```

真实GLM公开题B3/B4均自然复现：首次API响应完整，随后各11次CC非流式请求
因旧非流式路径对tools二次转换而报`KeyError: name`，未再次到达API。
真实Qwen响应回放也确认同一个thinking/text问题；它不是模型没输出答案。

已实现的转换修复：

- 按内容类型关闭旧块、分配新index，保留思考和正文，兼容两种reasoning字段。
- 拼完整分片工具名称、ID和JSON参数；验证完整后才交付工具块，参数可能缓冲。
  无效/缺失参数不能补成空对象假装成功，不完整工具调用明确失败。
- force模式下，下游JSON请求也使用同一条上游流式转换、验证、聚合路径，避免旧
  非流式tools重复转换分支。仅给原版添加环境变量不能获得这项修复。
- 将历史中的system消息合并到开头，保留角色和内容，避免Qwen模板因system位置报400。
- 中止流时显式关闭SDK内层HTTP迭代器，释放连接，避免遗留连接和后续请求卡住。

实现入口：[api_proxy.py](../ai4sci_bench/adapters/api_proxy.py)。

## 2. 启用配置

以下是固定CC 2.1.268下验证过的诊断起点，不是所有模型的统一最优预算。
在启动`asibench`的同一shell/作业环境设置，代理读取ASIBENCH变量，adapter将
允许的CC变量显式传入OS容器。API key继续通过agent配置的`api_key_env`提供。

公共等待与客户端控制：

```bash
export ASIBENCH_STREAM_IDLE_TIMEOUT_SECONDS=600
export API_TIMEOUT_MS=3600000
export ANTHROPIC_MAX_RETRIES=0
export CLAUDE_CODE_MAX_RETRIES=0
export CLAUDE_CODE_RETRY_WATCHDOG=0
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_ENABLE_STREAM_WATCHDOG=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000
export CLAUDE_ENABLE_BYTE_WATCHDOG=1
export CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS=600000
export API_FORCE_IDLE_TIMEOUT=0
```

OpenAI-compatible路径（`api_protocol=openai`，base按服务要求包含`/v1`）：

```bash
export ASIBENCH_FORCE_UPSTREAM_STREAMING=1
export ASIBENCH_STRICT_STREAM_ATTEMPTS=1
export ASIBENCH_STREAM_RETRY_ATTEMPTS=1
```

| 参数 | 默认及作用 | 范围与限制 |
| --- | --- | --- |
| `ASIBENCH_FORCE_UPSTREAM_STREAMING` | 默认0；设1强制该bridge向模型请求stream=true | 覆盖旧`ASIBENCH_REAL_STREAMING_PROXY=0`；CC要JSON时收齐并验证后聚合；不覆盖TokenRouter特殊路由和原生直连 |
| `ASIBENCH_STRICT_STREAM_ATTEMPTS` | 默认0；设1需要force=1及合法CC session UUID | 同session串行处理，失败后拦截后续隐式上游生成；正常工具续轮允许；显式恢复须新session并记录attempt |
| `ASIBENCH_STREAM_RETRY_ATTEMPTS` | 默认3；严格诊断设1 | 1表示最多一次代理尝试，不是重试一次；大于1也只允许符合条件的早期失败重试 |

原生Anthropic路径（`api_protocol=anthropic`，使用服务的Messages base）：

```bash
export ASIBENCH_NATIVE_STREAM_GUARD=1
```

Native guard默认关闭，启用后保留原生body、应用协议headers和thinking签名，认证与
连接相关HTTP头按转发需要重写；逐事件转发并
验证；只放行stream=true，CC尝试false会在访问供应商前被拒绝。HTTP/协议失败
关闭session，也拦截完整空回复后的隐藏补答。需显式provider key及CC session UUID。
支持text/thinking/redacted_thinking/tool_use；未知块明确报错，原生server tools须另验。
OpenAI的force/strict不能代替native guard，TokenRouter特殊路由也不在其范围内。

仅关闭CC retry/fallback参数不够：受控实验中404仍可触发特殊fallback，完整空回复
仍可追加补答提示。尝试边界由代理保护。状态仅保存在代理内存中，重启会清空；
同session辅助请求共享失败状态。启用force要求上游实际支持流式。

## 3. 空回复、断流和重采样

| 原始观测 | 判断 |
| --- | --- |
| 正常stop/end_turn并合法结束，无正文/工具 | 完整空回复，不自动归为基础设施失败 |
| 只有thinking，真实finish_reason=length | 提供方报告长度限制，不能说是网络断流 |
| 未见有效终态就EOF/断开 | 接收不完整，不能把SDK补出的stop当真实完成 |
| HTTP200但SSE含error | 上游服务错误，不能只看HTTP状态 |
| content_filter | 上游报告过滤；仅凭网关返回无法定位更内层原因 |
| 有合法工具调用，无正文 | 正常工具轮次，不算空生成 |
| CLI最终文字为空，但已有文件 | 按任务契约检查；反过来有文字也不保证产物存在 |

严格OpenAI模式及native guard保留第一次完整空回复，然后识别固定CC版本的标记
`[Your previous response had no visible output. Please continue and produce a user-visible response.]`
并在第二次调用API前阻止隐式补答。正常问题、工具轮次、历史消息中的旧标记不会因此
误拦。max_tokens后的自动续写是另一条分支，本轮未宣称所有继续行为均被禁用。

OpenAI桥接在LiteLLM归一化之前验证真实finish_reason，拒绝流内error、非法事件、
finish后新内容和无finish EOF。锁定SDK会在部分EOF路径自动补stop，回归测试覆盖了
这个问题。真实finish后正常EOF的支持路径不强制要求可选DONE尾标；只有DONE而无
finish不能算成功。Native要求块闭合、stop_reason和message_stop，合法message_stop
后不继续等待HTTP关闭。已知HTTP错误读取诊断body最多另等2秒，避免错误正文卡住。

实现：[native_anthropic_guard.py](../ai4sci_bench/adapters/native_anthropic_guard.py)。

## 4. timeout各层分别设置

| 层 | 参数 | 单位与边界 |
| --- | --- | --- |
| 代理到模型的流式等待 | `ASIBENCH_STREAM_IDLE_TIMEOUT_SECONDS` | 秒，默认600，网络读等待，不是整条生成总时长 |
| 旧阻塞路径等待 | `ASIBENCH_BLOCKING_TIMEOUT_SECONDS` | 秒，默认3600；force模式使用上面的流式预算 |
| CC请求 | `API_TIMEOUT_MS` | 毫秒，示例3600000；不覆盖其他watchdog及整题截止 |
| CC流/字节watchdog | `CLAUDE_STREAM_IDLE_TIMEOUT_MS`、`CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS`及enable变量 | 毫秒，示例600000；实际语义依赖固定CC版本 |
| 其他CC等待 | `CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS`、`CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS`、`MCP_TIMEOUT`、`MCP_TOOL_TIMEOUT` | 已显式透传；未逐一证明所有版本都支持或按字面阈值触发 |
| 输出/思考预算 | `CLAUDE_CODE_MAX_OUTPUT_TOKENS`、`MAX_THINKING_TOKENS` | token预算不是timeout；按模型和统一评测协议预先固定 |
| 整题总预算 | `--timeout` | 秒，仓库默认10800；Docker外层另有30秒清理宽限 |

旧版未把相关变量传给CC容器，宿主机设置可能无效。受控315秒静默实验中，旧链路
约302秒放弃并fallback；修复透传、应用整组等待设置后约316.94秒拿到原答案。
这验证的是整组配置，不能单独归因于一个timer。变量透传也不保证未来CC支持它。

流式不能消除首字节等待或整题超时。保持有限总预算；不要把本轮900秒/20次调用等
诊断限制当作正式标准。记录镜像、CLI/SDK、生效参数与恢复规则；改变预算应新建
实验组并保留旧结果，不按评分挑选重跑。

## 5. 采集和执行完成判断

- 有竞争读取的subprocess路径改为单一文件捕获，实时日志与最终结果消费同一完整
  字节源；各attempt用独立文件。OS路径有自己的采集方式，不能套用同一个丢字节根因。
- CC需合法最终result及显式成功字段；缺result、异常JSONL或未结束轮次不能只因
  exit0就判completed。代理记录的失败/未完成状态可推翻表面的CLI成功。
- 超时使用结构化异常判断；正常退出、超时和中断时清理POSIX同组后代，再读取
  最终证据。主动setsid脱离进程组及Windows整棵树仍需容器/cgroup等保障。
- HTTP错误保留状态；流失败明确报错，不把半截回复以成功终态收尾。

实现：[claude_code_cli.py](../ai4sci_bench/adapters/claude_code_cli.py)、
[proc_util.py](../ai4sci_bench/runner/proc_util.py)、
[subprocess_base.py](../ai4sci_bench/adapters/subprocess_base.py)、
[os_sandbox.py](../ai4sci_bench/runner/os_sandbox.py)。

## 6. 截断定责：已实现与后续设计

现有保护检查真实终态、转换后块结构及CC最终result。诊断实验还单独保存了两跳
原始捕获；**普通运行并未自动获得实验的完整捕获台账**。

```text
A：模型/网关原始响应 → 转换器 → B：发给CC的响应 → C：最终结果与产物
```

A完整有内容而B非法，定位转换器；A/B完整而C缺失，查CC/采集/任务截止；A缺终态
只证明接收未完成，不能证明服务端推理还没完成。缺响应文件表示观测不足；HTTP正常
EOF也不证明正常生成结束。心跳只说明连接活动，不证明推理进展。

**后续观测设计，尚未随本次代码实现：** 持久化统一attempt/request ID、两跳首末
字节/模型增量时刻和原始finish_reason；取消前先写发起组件、原因与阈值，再关闭
连接，避免把后来的BrokenPipe当根因。当前bridge会归一化stop_reason，不能仅凭
下游stop_sequence反推原始content_filter，需要原始响应证据。Qwen服务端还需关联
queued、generation_started、finished、aborted及输出hash；商业API需要供应商请求ID
和网关日志。拿不到服务端状态时保留unknown。这些不是已经部署的服务端插桩。

重新调用成功不证明第一次被截断；同一请求继续观察应单设诊断协议，后到答案不能
偷偷补入原样本成绩。评测应分开记录协议完整性、停止原因、内容类型、任务是否完成。

## 7. 验证与复现

机制实验固定CC 2.1.268、LiteLLM 1.82.6、OpenAI SDK 2.29.0、HTTPX 0.28.1。

| 验证 | 结果与范围 |
| --- | --- |
| 两套真实Qwen服务小任务 | 4/4任务产物符合简单预期，26/26请求流式；不是正式bench分数 |
| 真实OpenAI公开题修复版 | 同一道题B3/B4、两组10次运行，140次请求全stream=true；14次思考→正文无错块 |
| 最终真实CC故障对照 | 9/9用例、56/56断言符合预期；包含原版空成功对照，不是9个真实模型任务全成功 |
| Native真实CC矩阵 | 最终28项符合预期；真实Sonnet保护验收5次流式调用完成CSV任务 |
| 空回复保护回归 | 当时Linux相关181项通过；本次发布扩大到下列8个文件，共191项通过 |
| 本次发布全套离线测试 | 2518通过、2失败、2跳过、22未选；两项失败在未修改483f6a3上原样复现 |

公开题使用固定诊断镜像，未评分，不是全题库；原版/修复版prompt前缀有差异。
修复组仍有length、过滤、服务错误、整题/调用预算耗尽，不能宣称所有任务都成功或
推断准确率提升幅度。本次两项既有失败分别是Codex测试需要环境中缺少的uv，以及
Codex工具隔离测试仍断言旧的--sandbox参数；本次未修改Codex逻辑，全套不是全绿。

离线回归，不调用真实模型：

```bash
uv run pytest -q tests/test_claude_proxy_integrity.py \
  tests/test_claude_litellm_wire.py tests/test_claude_strict_attempts.py \
  tests/test_claude_empty_recovery.py tests/test_native_anthropic_guard.py \
  tests/test_capture_integrity.py tests/test_cc_review_edges.py \
  tests/test_claude_runtime_integrity.py
```

首次失败保留，恢复按预声明规则新建attempt。完整空生成和length不能自动从分母
剔除。分别报告任务正确率、完成率、传输失败原因、原始尝试与恢复次数。
