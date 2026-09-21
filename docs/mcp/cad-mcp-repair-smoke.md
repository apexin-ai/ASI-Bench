# CAD MCP 三项兼容性修复与复测

日期：2026-09-21。针对 2026-09-20 首轮报告中 SketchUp、Fusion 360、CadQuery
三项失败，修复固定版本的独立本地副本。不是上游已发布修复，不更改旧报告的历史结果。
本次不更新 catalog、不修改含凭据的 `mcp-config.json`，不连接真实 CAD/设备。

## 结果

| 实现 | 修复 | 实测结果 | 尚未验证 |
|---|---|---|---|
| SketchUp | FastMCP 的 `description` 改为 `instructions`；SDK 固定 1.30.0 | initialize、10 个工具、分页终止/重复列表一致性、ping、未知工具错误及后续响应通过 | SketchUp 宿主、插件、建模业务 |
| Fusion 360 | `--mcp` 改用 SDK stdio Server，转换 registry schema，保留生成器 | initialize、10 个工具、重复列表、ping、未知工具和错误类型参数；CreateSketch 生成含 adsk 调用的 Python 源码通过 | Fusion 安装、许可证及脚本执行/产物正确性 |
| CadQuery | 修复 `src.*` 导入；移除冗余 root_validator，使用 Pydantic v2 schema；标准 MCP stdio 分发到现有 handlers | initialize、10 个工具、重复列表、ping、空索引查询、缺必填参数/不存在结果的错误及错误后继续响应；安装包模型验证通过 | 几何建模、STEP/STL 导出、工作区依赖安装；旧 HTTP/SSE 不是本次标准 MCP 修复范围 |

三项均协商协议版本 `2025-11-25`。这里只表示本地 patched 版本通过 L0 和有限安全
调用；不能直接把首轮总表改为官方上游通过或业务可用。

## 红→绿及额外检查

- 项目相关基线：`uv run --frozen pytest tests/test_mcp_config.py -q`，11 passed。
  这是当前 origin/main 的测试数量，不是另一 worktree 报告中的 15 项。
- 新增 live 回归在未修复独立安装环境中：**4 failed**，覆盖三个 stdio 入口和
  CadQuery Pydantic 模型导入/验证。
- 修复后同组回归：**4 passed**；固定三个工具数量均为 10。
- CadQuery 冷加载本地库曾超过首版测试的 15 秒限时；增加为有界 60 秒后通过，
  最终完整 live 测试耗时约 11 秒。没有以跳过或接受启动失败替代通过。
- 三份干净固定 revision 重放补丁，输出 diff 与保存的补丁逐字节一致。
- 三个独立环境分别执行 `uv pip check`，全部兼容。
- 项目离线补丁验证及 MCP 配置测试命令：
  `uv run --frozen pytest tests/test_cad_mcp_repairs.py tests/test_mcp_config.py -q`，**16 passed**。

## 复现与本机状态

补丁、源 revision、SHA-256、实测依赖快照和执行步骤保存在
`scripts/mcp/cad-repairs/`。不会把上游源码或 CAD 软件捆绑进框架发行物。

- 旧安装保留：`$HOME/.local/share/asibench/cad-mcp-smoke/`。
- 修复安装：`$HOME/.local/share/asibench/cad-mcp-repaired/{sketchup,fusion360,cadquery}/`。
- 原始红/绿日志：修复安装根目录的 `before.log`、`after.log`。
- 临时 HOME、临时 cwd、无额外 PYTHONPATH；SketchUp 固定端口在测试中被本地
  非监听 socket 占用保护。如果端口已有服务，测试直接失败，不连接真实插件。
- 未执行 CAD 建模、任意脚本、文件导出、包安装工具、Ruby、GUI 或设备控制。
  修复并不提供执行沙箱；实际业务工具仍须单独审计、授权和隔离。
