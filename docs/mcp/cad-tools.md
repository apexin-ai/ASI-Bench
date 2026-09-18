# CAD 相关 MCP catalog 条目清单

> 本文是仓库配置的静态筛选与归类，不是已安装 MCP 服务清单，也不是成功调用报告。

## 1. 范围与统计

### 重叠说明与计数口径

**59 个 catalog ID 不代表 59 个独立软件，也不代表 59 组互不重合的 MCP 功能。** 需要区分以下三种情况：

| 重叠层次 | 当前结论 | 统计影响 |
|---|---|---|
| 同一对象的不同名称 | `open_cascade` / `occt`、`freecad_path` / `freecad_cam`、`usd` / `openusd` 共 3 组，涉及 6 个条目；分别对应项目全称与简称、工作台新旧名称、同一 USD/OpenUSD 项目 | 仅合并这三组，59 个目录条目可归一为 56 个目标条目；不代表其余条目均为独立软件 |
| 不同软件的相似能力 | 不同目标可能覆盖相似的草图、三维建模、几何交换或制造操作；功能相似不等于同一软件，也不意味着可以直接互相替代 | 不能仅凭 category、名称或软件介绍计算重复功能数 |
| MCP 实际接口重合 | catalog 没有提供各服务完整的 tool 清单、输入输出 schema 和成功调用记录 | 暂时无法计算已暴露 MCP 功能的重合数量或比例；上游软件能力不能直接视为 MCP 已接入能力 |

本文保留全部原始 ID 以便追溯，不在清单中静默合并。后续应分别核验实体归一关系和“条目 × 能力”矩阵，并区分上游软件具备、MCP 已暴露与实际调用验证通过三个状态。

### 原始条目统计

- 整理日期：2026-09-18。
- 输入：`ai4sci_bench/data/` 下四份 `science_mcp_catalog*.json`，共 510 个条目 ID。
- 本文完整覆盖 `cad`、`cad-cam`、`3d-modeling` 三类，并补充部分电子版图、设计数据管理、CAE 可视化和三维场景邻接项。
- 邻接项属于人工筛选口径；未纳入的一般仿真求解器、机器人、工业协议、MES/ERP 等，不代表不能用于 CAD 工作流。
- 共收录 **59 个 catalog ID**：其中 **40 个**来自上述三个完整类别，**19 个**是邻接项。这不是去重后的软件数量。
- 接入状态：**49 个明确标为 `candidate_wrapper`**，**10 个未标注**；未标注不等于已实现或已验证。

| 来源文件 | 本文收录数 |
|---|---:|
| `science_mcp_catalog.json` | 5 |
| `science_mcp_catalog_extended.json` | 4 |
| `science_mcp_catalog_additional.json` | 1 |
| `science_mcp_catalog_additional2.json` | 49 |

### 状态与来源阅读说明

- **候选**：原始字段为 `availability: candidate_wrapper`。
- **未标注**：原始条目没有 `availability` 字段，不能推断可用性。
- 每个条目的原始来源保留在后面的明细中。GitHub 搜索页不是已确认的 MCP 实现；软件本体仓库也不等于 MCP bridge。
- 启动配置中的命令名及 `/path/to/` 只是原始模板，不保证对应命令存在。本文不安装软件、不启动第三方服务。

## 2. 分类索引

### 2.1 核心 CAD 与通用三维建模（6 项）

收录 catalog 的全部 `cad` 和 `3d-modeling` 条目；三维建模作为 CAD 邻接范围单独说明。

| 条目 ID | 原始 category | 来源文件 | 状态 |
|---|---|---|---|
| `freecad` | `cad` | `science_mcp_catalog.json` | 未标注 |
| `autocad` | `cad` | `science_mcp_catalog.json` | 未标注 |
| `fusion360` | `cad` | `science_mcp_catalog.json` | 未标注 |
| `sketchup` | `cad` | `science_mcp_catalog.json` | 未标注 |
| `openscad` | `cad` | `science_mcp_catalog_extended.json` | 未标注 |
| `blender` | `3d-modeling` | `science_mcp_catalog.json` | 未标注 |

### 2.2 CAD 产品、几何内核与程序化建模（17 项）

从 `cad-cam` 类别中划分出的 CAD/几何相关目标。

| 条目 ID | 原始 category | 来源文件 | 状态 |
|---|---|---|---|
| `onshape` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `solidworks` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `siemens_nx` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `catia` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `ptc_creo` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `solid_edge` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `rhino3d` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `grasshopper` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `open_cascade` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `occt` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `salome_geometry` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `brlcad` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `cadquery` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `build123d` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `solvespace` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `librecad` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `autodesk_inventor` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |

### 2.3 CAM、数控与增材制造配套（15 项）

保留 `cad-cam` 类别的制造侧条目，但不将其计作独立 CAD 软件。

| 条目 ID | 原始 category | 来源文件 | 状态 |
|---|---|---|---|
| `freecad_path` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `freecad_cam` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `linuxcnc` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `machinekit` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `grbl` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `fluidnc` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `marlin` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `klipper` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `prusaslicer` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `curaengine` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `orcaslicer` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `camotics` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `opencamlib` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `pycam` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `bcnc` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |

### 2.4 电子 CAD / 版图设计邻接项（5 项）

电子设计与机械 CAD 分开列示；不是全部 EDA 工具清单。

| 条目 ID | 原始 category | 来源文件 | 状态 |
|---|---|---|---|
| `kicad` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `altium_designer` | `cad-cam` | `science_mcp_catalog_additional2.json` | 候选 |
| `openroad` | `electronic-design-automation` | `science_mcp_catalog_extended.json` | 未标注 |
| `klayout` | `electronic-design-automation` | `science_mcp_catalog_extended.json` | 未标注 |
| `magic_vlsi` | `eda-layout` | `science_mcp_catalog_additional.json` | 未标注 |

### 2.5 设计数据管理与 PLM 配套（8 项）

从 `mes-plm` 中挑选设计/产品数据管理相关目标；不纳入一般 MES、ERP 和生产调度。

| 条目 ID | 原始 category | 来源文件 | 状态 |
|---|---|---|---|
| `teamcenter` | `mes-plm` | `science_mcp_catalog_additional2.json` | 候选 |
| `ptc_windchill` | `mes-plm` | `science_mcp_catalog_additional2.json` | 候选 |
| `enovia_3dexperience` | `mes-plm` | `science_mcp_catalog_additional2.json` | 候选 |
| `aras_innovator` | `mes-plm` | `science_mcp_catalog_additional2.json` | 候选 |
| `fusion_manage` | `mes-plm` | `science_mcp_catalog_additional2.json` | 候选 |
| `solidworks_pdm` | `mes-plm` | `science_mcp_catalog_additional2.json` | 候选 |
| `autodesk_vault` | `mes-plm` | `science_mcp_catalog_additional2.json` | 候选 |
| `openbom` | `mes-plm` | `science_mcp_catalog_additional2.json` | 候选 |

### 2.6 CAE 可视化、平台与三维场景配套（8 项）

仅作宽口径 CAD 工作流邻接项，不认定为核心 CAD，也不穷举全部求解器或数字孪生平台。

| 条目 ID | 原始 category | 来源文件 | 状态 |
|---|---|---|---|
| `paraview` | `visualization` | `science_mcp_catalog_extended.json` | 未标注 |
| `calculix_graphix` | `cae-hpc` | `science_mcp_catalog_additional2.json` | 候选 |
| `febiostudio` | `cae-hpc` | `science_mcp_catalog_additional2.json` | 候选 |
| `salome_meca` | `cae-hpc` | `science_mcp_catalog_additional2.json` | 候选 |
| `simscale_api` | `cae-hpc` | `science_mcp_catalog_additional2.json` | 候选 |
| `nvidia_omniverse` | `digital-twin-monitoring` | `science_mcp_catalog_additional2.json` | 候选 |
| `usd` | `digital-twin-monitoring` | `science_mcp_catalog_additional2.json` | 候选 |
| `openusd` | `digital-twin-monitoring` | `science_mcp_catalog_additional2.json` | 候选 |

## 3. 重叠与计数注意事项

- **不同 ID 不保证独立实体**：`open_cascade` / `occt`、`usd` / `openusd` 是需要优先做实体归一核验的名称组。原始配置没有 canonical ID，本文保留原 ID，不据此给出独立软件总数。
- **软件与模块混列**：`freecad`、`freecad_path`、`freecad_cam` 分开收录；整理时应核验后两项是否存在工作台新旧命名关系，不应直接按三款软件计数。
- **产品与配套分列**：如 `solidworks` / `solidworks_pdm`、`fusion360` / `fusion_manage`，属于需要区分角色的关联项，不能仅因名称相近就删除。
- **同属 CAD 工作流不等于相同工具**：几何内核、CAD 应用、CAM、切片、数控控制、PLM 与后处理分别承担不同角色。

## 4. 原始来源、前置条件与启动配置

以下字段从对应 JSON 原样提取，方便后续逐项核验。来源地址以代码展示，候选搜索地址不代表已确认的上游项目。

### 核心 CAD 与通用三维建模

#### `freecad`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog.json`
- 原始来源：`https://github.com/bonninr/freecad_mcp`
- availability：`未标注`
- 原始前置条件：
  - Install FreeCAD
  - Install the freecad_mcp addon
  - Replace the Python and bridge paths
- 原始 server 配置：

```json
{
  "command": "/path/to/python",
  "args": [
    "/path/to/freecad_mcp/src/freecad_bridge.py"
  ]
}
```

#### `autocad`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog.json`
- 原始来源：`https://github.com/U-C4N/Autocad-MCP`
- availability：`未标注`
- 原始前置条件：
  - Install autocad-mcp-pro
  - Set ALLOWED_PATHS
  - Use ezdxf headless or install AutoCAD for COM mode
- 原始 server 配置：

```json
{
  "command": "autocad-mcp",
  "args": [],
  "env": {
    "AUTOCAD_MCP_BACKEND": "auto",
    "ALLOWED_PATHS": "/path/to/allowed/cad/files",
    "DISCOVERY_MODE": "search"
  }
}
```

#### `fusion360`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog.json`
- 原始来源：`https://github.com/ArchimedesCrypto/fusion360-mcp-server`
- availability：`未标注`
- 原始前置条件：
  - Install Autodesk Fusion 360
  - Clone and install the server
  - Configure the Fusion add-in/bridge
- 原始 server 配置：

```json
{
  "command": "/path/to/python",
  "args": [
    "/path/to/fusion360-mcp-server/src/main.py",
    "--mcp"
  ]
}
```

#### `sketchup`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog.json`
- 原始来源：`https://github.com/BearNetwork-BRNKC/SketchUp-MCP`
- availability：`未标注`
- 原始前置条件：
  - Install SketchUp
  - Install and enable its MCP extension
  - Install uv
- 原始 server 配置：

```json
{
  "command": "uvx",
  "args": [
    "sketchup-mcp"
  ]
}
```

#### `openscad`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_extended.json`
- 原始来源：`https://github.com/openscad/openscad`
- availability：`未标注`
- 原始前置条件：
  - Install OpenSCAD
  - Install an OpenSCAD MCP server
- 原始 server 配置：

```json
{
  "command": "openscad-mcp",
  "args": []
}
```

#### `blender`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog.json`
- 原始来源：`https://github.com/ahujasid/blender-mcp`
- availability：`未标注`
- 原始前置条件：
  - Install Blender 3.0+
  - Install uv
  - Install and enable the Blender MCP addon
- 原始 server 配置：

```json
{
  "command": "uvx",
  "args": [
    "blender-mcp"
  ]
}
```

### CAD 产品、几何内核与程序化建模

#### `onshape`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=onshape+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure onshape
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "onshape-mcp",
  "args": []
}
```

#### `solidworks`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=solidworks+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure solidworks
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "solidworks-mcp",
  "args": []
}
```

#### `siemens_nx`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=siemens_nx+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure siemens_nx
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "siemens_nx-mcp",
  "args": []
}
```

#### `catia`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=catia+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure catia
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "catia-mcp",
  "args": []
}
```

#### `ptc_creo`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=ptc_creo+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure ptc_creo
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "ptc_creo-mcp",
  "args": []
}
```

#### `solid_edge`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=solid_edge+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure solid_edge
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "solid_edge-mcp",
  "args": []
}
```

#### `rhino3d`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=rhino3d+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure rhino3d
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "rhino3d-mcp",
  "args": []
}
```

#### `grasshopper`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=grasshopper+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure grasshopper
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "grasshopper-mcp",
  "args": []
}
```

#### `open_cascade`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=open_cascade+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure open_cascade
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "open_cascade-mcp",
  "args": []
}
```

#### `occt`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=occt+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure occt
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "occt-mcp",
  "args": []
}
```

#### `salome_geometry`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=salome_geometry+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure salome_geometry
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "salome_geometry-mcp",
  "args": []
}
```

#### `brlcad`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=brlcad+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure brlcad
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "brlcad-mcp",
  "args": []
}
```

#### `cadquery`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=cadquery+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure cadquery
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "cadquery-mcp",
  "args": []
}
```

#### `build123d`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=build123d+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure build123d
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "build123d-mcp",
  "args": []
}
```

#### `solvespace`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=solvespace+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure solvespace
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "solvespace-mcp",
  "args": []
}
```

#### `librecad`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=librecad+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure librecad
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "librecad-mcp",
  "args": []
}
```

#### `autodesk_inventor`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=autodesk_inventor+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure autodesk_inventor
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "autodesk_inventor-mcp",
  "args": []
}
```

### CAM、数控与增材制造配套

#### `freecad_path`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=freecad_path+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure freecad_path
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "freecad_path-mcp",
  "args": []
}
```

#### `freecad_cam`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=freecad_cam+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure freecad_cam
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "freecad_cam-mcp",
  "args": []
}
```

#### `linuxcnc`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=linuxcnc+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure linuxcnc
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "linuxcnc-mcp",
  "args": []
}
```

#### `machinekit`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=machinekit+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure machinekit
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "machinekit-mcp",
  "args": []
}
```

#### `grbl`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=grbl+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure grbl
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "grbl-mcp",
  "args": []
}
```

#### `fluidnc`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=fluidnc+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure fluidnc
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "fluidnc-mcp",
  "args": []
}
```

#### `marlin`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=marlin+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure marlin
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "marlin-mcp",
  "args": []
}
```

#### `klipper`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=klipper+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure klipper
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "klipper-mcp",
  "args": []
}
```

#### `prusaslicer`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=prusaslicer+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure prusaslicer
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "prusaslicer-mcp",
  "args": []
}
```

#### `curaengine`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=curaengine+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure curaengine
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "curaengine-mcp",
  "args": []
}
```

#### `orcaslicer`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=orcaslicer+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure orcaslicer
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "orcaslicer-mcp",
  "args": []
}
```

#### `camotics`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=camotics+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure camotics
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "camotics-mcp",
  "args": []
}
```

#### `opencamlib`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=opencamlib+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure opencamlib
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "opencamlib-mcp",
  "args": []
}
```

#### `pycam`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=pycam+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure pycam
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "pycam-mcp",
  "args": []
}
```

#### `bcnc`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=bcnc+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure bcnc
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "bcnc-mcp",
  "args": []
}
```

### 电子 CAD / 版图设计邻接项

#### `kicad`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=kicad+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure kicad
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "kicad-mcp",
  "args": []
}
```

#### `altium_designer`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=altium_designer+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure altium_designer
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "altium_designer-mcp",
  "args": []
}
```

#### `openroad`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_extended.json`
- 原始来源：`https://github.com/The-OpenROAD-Project/OpenROAD`
- availability：`未标注`
- 原始前置条件：
  - Install OpenROAD and PDK files
  - Install an OpenROAD MCP bridge
- 原始 server 配置：

```json
{
  "command": "openroad-mcp",
  "args": []
}
```

#### `klayout`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_extended.json`
- 原始来源：`https://github.com/KLayout/klayout`
- availability：`未标注`
- 原始前置条件：
  - Install KLayout
  - Install a KLayout Python/Ruby MCP bridge
- 原始 server 配置：

```json
{
  "command": "klayout-mcp",
  "args": []
}
```

#### `magic_vlsi`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional.json`
- 原始来源：`https://opencircuitdesign.com/magic/`
- availability：`未标注`
- 原始前置条件：
  - Install Magic VLSI
  - Install a Magic MCP bridge
- 原始 server 配置：

```json
{
  "command": "magic-mcp",
  "args": []
}
```

### 设计数据管理与 PLM 配套

#### `teamcenter`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=teamcenter+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure teamcenter
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "teamcenter-mcp",
  "args": []
}
```

#### `ptc_windchill`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=ptc_windchill+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure ptc_windchill
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "ptc_windchill-mcp",
  "args": []
}
```

#### `enovia_3dexperience`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=enovia_3dexperience+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure enovia_3dexperience
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "enovia_3dexperience-mcp",
  "args": []
}
```

#### `aras_innovator`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=aras_innovator+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure aras_innovator
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "aras_innovator-mcp",
  "args": []
}
```

#### `fusion_manage`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=fusion_manage+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure fusion_manage
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "fusion_manage-mcp",
  "args": []
}
```

#### `solidworks_pdm`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=solidworks_pdm+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure solidworks_pdm
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "solidworks_pdm-mcp",
  "args": []
}
```

#### `autodesk_vault`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=autodesk_vault+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure autodesk_vault
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "autodesk_vault-mcp",
  "args": []
}
```

#### `openbom`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=openbom+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure openbom
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "openbom-mcp",
  "args": []
}
```

### CAE 可视化、平台与三维场景配套

#### `paraview`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_extended.json`
- 原始来源：`https://www.paraview.org/`
- availability：`未标注`
- 原始前置条件：
  - Install ParaView or pvpython
  - Enable a ParaView MCP/Trame bridge
- 原始 server 配置：

```json
{
  "command": "pvpython",
  "args": [
    "/path/to/paraview-mcp/server.py"
  ]
}
```

#### `calculix_graphix`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=calculix_graphix+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure calculix_graphix
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "calculix_graphix-mcp",
  "args": []
}
```

#### `febiostudio`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=febiostudio+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure febiostudio
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "febiostudio-mcp",
  "args": []
}
```

#### `salome_meca`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=salome_meca+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure salome_meca
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "salome_meca-mcp",
  "args": []
}
```

#### `simscale_api`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=simscale_api+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure simscale_api
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "simscale_api-mcp",
  "args": []
}
```

#### `nvidia_omniverse`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=nvidia_omniverse+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure nvidia_omniverse
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "nvidia_omniverse-mcp",
  "args": []
}
```

#### `usd`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=usd+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure usd
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "usd-mcp",
  "args": []
}
```

#### `openusd`

- 来源文件：`ai4sci_bench/data/science_mcp_catalog_additional2.json`
- 原始来源：`https://github.com/search?q=openusd+MCP&type=repositories`
- availability：`candidate_wrapper`
- 原始前置条件：
  - Install or configure openusd
  - Install or implement an MCP bridge
- 原始 server 配置：

```json
{
  "command": "openusd-mcp",
  "args": []
}
```

## 5. 验证边界与后续检查

- `tests/test_mcp_config.py` 验证配置解析、目录结构、CLI 配置生成和隔离边界，不证明上游服务可运行。
- `TEST.md` 明确说明科学 MCP 测试离线运行，不启动第三方 server。
- `asibench mcp check` 仅检查明显的本地命令/模板问题；HTTP 配置不会做连通性测试。
- 本文未执行真实 MCP 握手、`tools/list` 或 `tools/call`。不能据此宣称任何一个条目已经联调成功。
- 后续真实验证应分别记录：确定的上游 MCP 实现与版本、软件与许可证前置条件、启动结果、工具发现结果、最小调用输入/输出、失败原因，以及测试时间与环境。
