# analysis/ — E-V1/E-V2 处理管线(Python)

与老师的 MATLAB 管线(`../second_new.m`, `../secondload.m`)互为校验。

## 运行(按序)

```bash
python3 ev2_calibrate.py         # 标定 (v_f, w, P) → out/params.json + 图
python3 ev1_pure_class.py        # 纯类极限验证(依赖 params.json)+ A 对比热图
python3 e1_classify.py           # 轨迹 caught/free 分类 + 事件提取 → out/e1/*.npz
python3 ev3_calibrate_kappa.py   # κ_c, κ_r Poisson 曝露 MLE → out/ev3_kappa.json + 图
python3 test_solver.py           # 求解器测试(6 项)
python3 ev4_compare.py --form lf # E-V4 前向仿真对照 → out/ev4/(--form af → out/ev4_af/)
python3 ev4_compare.py --form lf --qxi 2000  # E-V4b 容量帽 → out/ev4b_q2000/
python3 ev5_dispersion.py        # E-V5 色散关系 → out/ev5/(audit_dispersion.py 独立复核)
python3 ev5_waves.py             # E-V5 实测波谱(--selftest 合成波校验)
python3 ev5_sim_vs_data.py       # E-V5 闭环:同箱体模型-数据谱对比
```

## 文件

| 文件 | 作用 |
|---|---|
| `loader.py` | 读 `Second/*.mat`:场景参数解析(A≥100→/100)、密度/流量场、updown、超车流、轨迹(含受控车);`Scenario`/`Rep` 数据类 |
| `fd.py` | 三角形基本图 Q/D/S、自由支稳健拟合、拥堵支拟合(P 固定/自由)、ECC22 式 (7) 稳态平均 |
| `ev2_calibrate.py` | E-V2:A=3 密集集 → (v_f, w, P);尾部激波 RH 一致性检验 |
| `ev1_pure_class.py` | E-V1:纯 B_f 极限、队列内部 vs 共享拥堵支、A=1 vs A=10 热图 |
| `e1_classify.py` | E1:滞回 caught/free 分类、捕获/释放事件、分类计数场 |
| `ev3_calibrate_kappa.py` | E-V3:κ_c/κ_r Poisson 曝露 MLE(a·f 与 (a+s)·f、模型/经验 Δv 各两变体) |
| `solver.py` / `test_solver.py` / `audit_solver.py` | .tex §11 分裂格式求解器(min-flux CTM + exact reaction,CAV 外生轨迹)+ 测试 + 独立数值审计 |
| `ev4_compare.py` | E-V4:前向仿真 vs SUMO(重栅格化、ρ-RMSE/N_s-MAE/ω/e_s 四指标、全部对照图) |
| `out/` | `params.json`、`ev1_metrics.json`、`ev3_kappa.json`、`e1/*.npz`、`ev4/`、`ev4_af/`、图 |

## 约定

单位:ρ [veh/km]、q [veh/h]、速度 [km/h](m/s×3.6);双车道聚合。
时间线:CAV 100 s 进入、250–750 s 慢行(u_ξ)、1000 s 结束;场采样 Δt=10 s × Δx=100 m。
只用 `_True.mat`(TEND=100)。

## 已知数据事实(踩坑记录)

- `updown` 行序:ρ₋, q₋, ρ₊, q₊(上游=CAV 后方 2L 窗,下游=前方 L..3L 窗)。
- 上游测量是**双车道聚合**:右道排队 ≈u_ξ + 左道超车更快 → 中等 u_ξ 的稳态点是两个 FD 态的凸组合,落在凹 FD **内部**(ECC22 用上凹包络的原因);只有强瓶颈(ρ₋≥60)才采样到拥堵支本身。
- 移动瓶颈的**消散波 −w 不可直接观测**:释放后 ρ>阈值 的边界是随车平移的物质边界(+6~11 m/s),非运动学波;有限加速度(2.6 m/s²)还会在前方造成过渡聚集。
- 队列**尾部激波**(生长期)是干净的:实测斜率与 RH 预测中位偏差 1.3 km/h。
- `trajectory_vehicle_S_{n1}_...` 首数字恒为 3,与文件 A 无关;按后缀 (t0, rep) 解析。
