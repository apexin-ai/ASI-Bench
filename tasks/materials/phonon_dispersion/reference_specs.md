# Reference Implementation Specs

> 参考：Simon, "The Oxford Solid State Basics", Chapter 10 (The Diatomic Chain)

## 物理模型

N 个原子，质量全部相同为 m，弹簧常数交替排列 κ1, κ2, κ1, κ2, ...，单元胞含 2 个原子，晶格常数 a。k 在第一布里渊区 [0, π/a]。

**色散关系：**
```
ω²±(k) = (κ1+κ2)/m ± (1/m) × sqrt(κ1² + κ2² + 2κ1κ2·cos(k·a))
```

解析验证：

| 位置 | 声学支 | 光学支 |
|------|--------|--------|
| k=0 | 0 | sqrt(2(κ1+κ2)/m) ← ω_max |
| k=π/a | sqrt(2·min(κ1,κ2)/m) | sqrt(2·max(κ1,κ2)/m) |

**DOS**：在 [0, π/a] 均匀采样 5000 个 k 点，两个支共 10000 个频率，做归一化直方图（∫ DOS dω = 2，每支贡献 1）。归一化除以 `N_DENSE × d_omega`（N_DENSE = 5000，即每支 k 点数）。

**热容**：对同一套 10000 个频率，每个温度 T 求 Einstein 单模公式之和再除以 N_DENSE（5000），排除 ω < 1e-10 的零模。高温极限 Cv → 2（Dulong-Petit，每 k 点两个模式）。

## 常见错误

**1. 色散公式用错模型**

Simon Ch.10 是"质量相同，弹簧不同"（κ1, κ2, m）。另一种常见教材用"弹簧相同，质量不同"（m1, m2, C），公式不同，数值不匹配。

**2. acoustic/optical 行顺序错误**

Row 0 必须是声学支（k=0 时 ω=0），Row 1 是光学支。颠倒直接导致色散满分变 0 分。

**3. k 路径端点**

`np.linspace(0, π/a, n_kpoints)` 两端都包含，是正确的。用 `endpoint=False` 会少一个点导致 shape 不对。

**4. DOS 归一化错误**

正确：`dos = counts / (N_DENSE × d_omega)`，使 `∫ DOS dω = 2`（N_DENSE = 5000，每支 k 点数）。
常见错误 1：除以 `2×N_DENSE × d_omega`（全部频率数），使 `∫ DOS dω = 1`（概率密度归一化）。
常见错误 2：除以 n_bins 而非 (N_DENSE × d_omega)。

**5. 热容归一化错误**

正确：`Cv = sum(Einstein(ω_i)) / N_DENSE`（对全部 2×N_DENSE 个模式求和，除以 N_DENSE = 5000）。高温极限 Cv → 2。
常见错误 1：用 `mean`（除以 2×N_DENSE），高温极限 Cv → 1，差一个因子 2。
常见错误 2：用 `sum`（不除以任何数），随 k 网格密度线性变化，不可复现。

**6. 未排除零模式**

声学支在 k=0 时 ω=0，需排除 ω < 1e-10，否则低温下出现 NaN。

**7. 用 Debye 或 Einstein 近似替代全声子计算**

B4 提到了 Debye/Einstein 模型，agent 可能误用解析近似代替真正的声子求和，热容曲线形状会有偏差（15 分扣损）。

**8. omega_max 的 1.005 margin**

`params.json` 里的 `omega_max` 比真实最大频率略大（×1.005），确保最高频率落在直方图最后一个 bin 内。agent 应直接使用 `params.json` 中给定的 `omega_max`，不要自己重新计算。
