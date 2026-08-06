# Adv-BMT：FSM–HOCBF-QP 二维运行时保障插件

本扩展在 Adv-BMT 的评估阶段加入一个即插即用的二维运行时保障（Runtime Assurance，RTA）层。它不重新训练 TD3，不修改奖励函数、策略观测、场景生成器或 MetaDrive 原生环境，只在 `TD3.predict()` 与 `env.step()` 之间过滤转向和加减速动作。

当前默认方法为 `fsm_hocbf_qp`：有限状态机负责持续的机动决策，二维 HOCBF-QP 负责连续控制修正。当前 FSM 路径只使用当前时刻的安全函数、道路余量和航向误差，不进行候选动作滚动或有限时域运动预测。Git 标签 `v1` 保留了此前“三组方向性参考 + 三次 QP + 短时候选前视选择”的版本；旧插件 `predictive_escape_cbf2d` 仍可直接运行，用于消融和回退。

> 这是研究原型。FSM 解决的是模式持续性和候选枚举的工程表达问题；安全约束仍来自 HOCBF。当前实现不应被表述为对 MetaDrive 完整动力学的全局安全证明。

## 1. 执行位置

```text
场景观测
  -> TD3.predict()
  -> 原始动作 [steering, throttle/brake]
  -> fsm_hocbf_qp 插件
       -> 风险激活与安全裕度计算
       -> FSM 状态转移
       -> 生成当前状态的单个参考控制
       -> 一次二维 HOCBF-QP
  -> 安全动作 [steering, throttle/brake]
  -> MetaDrive env.step()
```

原始入口 `eval_td3_policy.py` 保持不变。带插件的独立入口为 `eval_td3_policy_rta.py`，通过 `--rta_plugin none` 可以在同一评估脚本中关闭 RTA。

## 2. FSM–HOCBF-QP 方法

### 2.1 控制量与运动学近似

插件把 MetaDrive 的归一化动作转换为物理控制量

\[
u=[a,\kappa]^\mathsf{T},
\]

其中 \(a\) 为纵向加速度，\(\kappa\) 为路径曲率。局部模型为

\[
\dot x=v\cos\theta,\quad \dot y=v\sin\theta,\quad
\dot v=a,\quad \dot\theta=v\kappa.
\]

模型只用于在线约束和参考控制计算；车辆最终状态仍由 MetaDrive 原生仿真推进。

### 2.2 FSM 状态

状态集合为：

- `nominal`：保持 TD3 参考控制，由 HOCBF-QP 进行必要的最小修正；
- `brake`：前向危险或没有可用侧向通道时使用风险与驻留时间自适应的制动参考；
- `escape_left`：保持向左避让的连续曲率参考；
- `escape_right`：保持向右避让的连续曲率参考；
- `recovery`：危险解除后进行航向恢复，再返回 `nominal`。

状态转移使用当前时刻的 HOCBF 指标 \(h\) 和 \(\psi_1\)、进入/退出双阈值、最小驻留步数、连续安全计数和恢复时长，避免控制周期之间反复左右切换。安全风险升级不受驻留计时器阻塞。

`brake` 不再固定执行 \(-1.5\,\mathrm{m/s^2}\)。先由当前前向屏障状态计算 \(r=\max([0.8-h]_+/0.8,[0.5-\psi_1]_+/0.5)\)，再与制动状态驻留超过 10 步后的 20 步线性增强项取最大值，得到 \(a_{\mathrm{ref}}=\min(a_{\mathrm{TD3}},-1.5r)\)。因此短时轻度风险只滑行或轻制动，持续风险才逐渐增强到最大制动；HOCBF-QP 仍负责最终硬安全修正。

左右方向不是两个固定动作。FSM 先依据当前道路剩余宽度、侧向交通和虚拟横向目标位置计算左右安全裕度，再选定一个持续状态；逃逸状态通过当前航向误差反馈产生一个连续曲率参考，随后只求解一次 QP。道路约束每步都计算用于遥测，但仅在当前 \(h\le 0.5\) 或 \(\psi_1\le 0\) 时进入 QP，降低远离边界时的误干预。

由于仅凭道路几何不能可靠识别 MetaDrive 的导航路线终止条件，默认配置关闭前向主动侧向逃逸；后向威胁的左右逃逸保持启用，前向威胁默认使用上述自适应制动。`fsm_front_escape_enabled` 仅作为后续路线可行性实验开关。

### 2.3 HOCBF-QP

车辆外形使用局部道路坐标系中的平滑超椭圆保守包络。前车、后车、侧向交通参与者和道路边界形成相对阶为二的 HOCBF 线性约束。每个控制周期求解

\[
u^*=\arg\min_u\|u-u^{\mathrm{ref}}_{q}\|_W^2,
\]

其中 \(q\) 是当前 FSM 状态。所有状态复用同一组车辆与道路安全约束，区别只在参考控制。硬约束没有公共解时，求解器优先保持道路约束，并记录车辆约束松弛和不可行程度。

### 2.4 与 v1 枚举版的区别

| 项目 | v1：`predictive_escape_cbf2d` | 当前：`fsm_hocbf_qp` |
|---|---|---|
| 离散结构 | 每步生成 nominal/left/right 三组候选 | 持续的五状态 FSM |
| 每个活跃控制步 | 三次 QP，随后分别短时滚动评分 | 当前状态只解一次 QP |
| 左右选择 | 每步重新比较候选 | 进入状态时选择，并通过滞回保持 |
| 短时预测 | 用于风险激活和候选动作评分 | 不使用；风险激活与参考控制均由当前状态计算 |
| 连续输出 | 是 | 是 |
| 用途 | 旧基线和消融 | 当前默认方法 |

FSM 仍然具有有限模式思想，因此不能宣称“完全消除了离散候选”；准确说法是：它取消了每个控制步的三候选并行求解和动作滚动选择，改为带记忆的监督状态机与单 QP 连续安全修正。

## 3. 目录

```text
eval_td3_policy.py                         原始评估入口，不启用 RTA
eval_td3_policy_rta.py                     可插拔 RTA 评估入口
diagnose_rta_scenario.py                   单场景逐步诊断
bmt/runtime_assurance/
├── fsm.py                                 FSM 监督器与单 QP HOCBF 过滤器
├── adaptive_plugin.py                     新旧 MetaDrive 插件入口
├── cbf2d.py                               v1 候选枚举 HOCBF 基线
├── metadrive_plugin.py                    状态、道路和动作适配器
├── qp.py                                  二维优先级 QP 求解器
├── metrics.py                             速度、终止口径与运行延迟指标
└── types.py                               配置、结果与安全遥测结构
cfgs/
├── rta_fsm_hocbf_default.json             当前 FSM 默认配置
└── rta_cbf2d_*.json                       v1 枚举版配置
tests/test_runtime_assurance_cbf2d.py      QP、HOCBF、FSM 与插件测试
```

## 4. 运行

### 4.1 当前 FSM 版本

```powershell
python eval_td3_policy_rta.py `
  --ckpt_path "td3_waymo_native\seed_0_300000_steps.zip" `
  --eval_data_dir "Adv-BMT scenarios\rear_attack_eval_100" `
  --num_episodes 100 `
  --horizon 100 `
  --start_index 0 `
  --device cpu `
  --rta_plugin fsm_hocbf_qp `
  --rta_config "cfgs\rta_fsm_hocbf_default.json" `
  --out "analysis_outputs\rta_diagnostics\my_fsm_rta.json"
```

`fsm_hocbf_qp` 已是两个 RTA 脚本的默认插件名，但论文实验建议显式写出插件和配置，保证可复现。

### 4.2 无 RTA 基线

```powershell
python eval_td3_policy_rta.py `
  --ckpt_path "td3_waymo_native\seed_0_300000_steps.zip" `
  --eval_data_dir "Adv-BMT scenarios\rear_attack_eval_100" `
  --num_episodes 100 `
  --horizon 100 `
  --device cpu `
  --rta_plugin none `
  --out "analysis_outputs\rta_diagnostics\my_no_rta.json"
```

### 4.3 v1 枚举消融

```powershell
python eval_td3_policy_rta.py `
  --ckpt_path "td3_waymo_native\seed_0_300000_steps.zip" `
  --eval_data_dir "Adv-BMT scenarios\rear_attack_eval_100" `
  --num_episodes 100 `
  --horizon 100 `
  --device cpu `
  --rta_plugin predictive_escape_cbf2d `
  --rta_config "cfgs\rta_cbf2d_side_vru_completion.json" `
  --out "analysis_outputs\rta_diagnostics\my_enumeration_rta.json"
```

## 5. 遥测

逐步遥测新增：

- `rta_mode`、`rta_fsm_previous_mode`：当前和上一 FSM 状态；
- `rta_fsm_transitioned`、`rta_fsm_transition_reason`：是否转移及原因；
- `rta_fsm_state_age_steps`：当前状态驻留步数；
- `rta_fsm_left_margin`、`rta_fsm_right_margin`：左右安全裕度；
- `rta_reference_acceleration/curvature`：当前状态参考控制；
- `rta_qp_solve_count`：本步 QP 调用次数；
- HOCBF 的 \(h\)、\(\dot h\)、\(\psi_1\)、残差和可行性；
- 原始控制、安全控制、活跃障碍物数量、回退类型和道路几何来源。

回合与汇总结果还包括互斥的 `termination_reason`、原始事件标志、参考路线长度、退化场景标志，以及仅在有效动态场景上计算的奖励、完成度、碰撞率、越界率和到达率。

## 6. 当前验证结果

同一个原生场景训练 TD3 检查点、相同 100 个追尾场景、确定性推理、100 步时域和 CPU：

| 方法 | 平均奖励 | 完成度 | 碰撞率 | 越界率 | 到达率 | 全步平均速度 |
|---|---:|---:|---:|---:|---:|---:|
| 无 RTA | 28.786 | 0.638 | 17% | 29% | 64% | 11.601 m/s |
| v17 枚举 HOCBF-QP | 33.321 | 0.693 | 5% | 28% | 73% | 10.564 m/s |
| 旧 FSM | 32.392 | 0.678 | 6% | 26% | 74% | 9.881 m/s |
| 修改前固定制动 FSM–HOCBF-QP | 34.562 | 0.623 | 7% | 18% | 75% | 9.626 m/s |
| 道路修正前的自适应制动 FSM–HOCBF-QP | 34.440 | 0.617 | 5% | 21% | 75% | 10.237 m/s |
| **当前道路感知 FSM–HOCBF-QP** | **38.272** | **0.661** | **5%** | **14%** | **83%** | **10.226 m/s** |

这里的碰撞、越界和到达是原始事件标志，可能重叠，不能直接相加。当前版本按 MetaDrive 终止优先级得到 83% 成功、10% 道路违规、5% 碰撞和 2% 超时或其他终止。相对道路修正前版本，越界事件率降低 7 个百分点、道路违规终止率降低 8 个百分点、成功率提高 8 个百分点，平均奖励提高 3.832、完成度提高 0.044；碰撞率保持 5%，平均速度只下降 0.011 m/s（0.11%）。

本次道路修正不改变 TD3、奖励函数或 MetaDrive 环境。插件对参考路线边界与侧向实线探测距离逐侧取较小值，防止物理探测结果错误放宽导航路线；道路安全函数同时使用车身矩形沿道路法向的实时投影

\[
e_\perp(\theta)=\frac{L}{2}|\sin(\theta-\theta_r)|+\frac{W}{2}|\cos(\theta-\theta_r)|,
\]

替代固定半车宽，从而约束高速转向时更容易先越线的车身角点。后向威胁判断也直接使用当前二维 HOCBF 状态，不再等待前后车辆完全横向重叠；该判断不包含有限时域轨迹预测。

### 6.1 有效动态场景口径

100 个场景中有 26 个参考路线长度小于 2 m，环境在单步内终止，其 `route_completion` 会在接近 0 和 1 之间跳变。排除这些退化场景后，同场景对比如下：

| 方法 | 场景数 | 平均奖励 | 完成度 | 碰撞率 | 越界率 | 到达率 |
|---|---:|---:|---:|---:|---:|---:|
| 无 RTA | 74 | 37.143 | 0.707 | 22.97% | 28.38% | 51.35% |
| v17 枚举 HOCBF-QP | 74 | 43.271 | 0.781 | **6.76%** | 28.38% | 63.51% |
| 旧 FSM | 74 | 42.016 | 0.773 | 8.11% | 24.32% | 64.86% |
| 修改前固定制动 FSM–HOCBF-QP | 74 | 44.949 | 0.780 | 9.46% | 20.27% | 66.22% |
| 道路修正前的自适应制动 FSM–HOCBF-QP | 74 | 44.784 | 0.771 | **6.76%** | 24.32% | 66.22% |
| **当前道路感知 FSM–HOCBF-QP** | 74 | **49.962** | **0.834** | **6.76%** | **13.51%** | **77.03%** |

相对道路修正前版本，当前方法在有效动态场景中的奖励提高 5.179、完成度提高 0.062、越界率降低 10.81 个百分点、到达率提高 10.81 个百分点，碰撞率不变。相对 v17 枚举，当前方法的奖励提高 6.691、完成度提高 0.053、越界率降低 14.87 个百分点、到达率提高 13.52 个百分点，碰撞率相同；全步平均速度比 v17 低 3.20%。

当前结果保存在 `analysis_outputs/rta_diagnostics/paired_100_fsm_route_footprint.json`。过滤均值为 21.68 ms、P95 为 61.94 ms；在 100 ms 控制周期下，截止期失约率为 2.80%。平均每回合求解 35.64 次 QP、发生 1.99 次 FSM 转移。安全、进度和速度目标已经达到，但求解器在不可行回退时存在约 400–500 ms 的长尾延迟，仍应作为下一步实时性优化项。

这组结果支持“通过当前状态 FSM 监督、航向相关车身包络与路线/实线保守融合，在不增加碰撞和几乎不降低速度的条件下降低越界”的结论。正式论文仍应报告多随机种子、置信区间、参数敏感性，以及道路边界融合和航向相关车身包络的消融实验。

## 7. 测试

```powershell
python -m pytest tests\test_runtime_assurance_cbf2d.py `
  tests\test_runtime_assurance_metrics.py -q
```

当前测试覆盖：二维 QP 投影、道路约束优先回退、威胁激活、前后夹击、侧车方向选择、FSM 驻留与滞回、恢复状态、插件复位、单步只求解一次 QP、MetaDrive 动作范围和运行延迟指标。

## 8. 局限与论文表述边界

- FSM 本身不是本文创新，适合表述为与二维 HOCBF-QP 联合构成的运行时保障架构；
- 左右安全裕度仍是基于局部几何与虚拟横向目标的在线判据，不是 HJ 可达集或全局轨迹规划器；
- 当前版本完全基于当前状态，因此不会提前响应当前安全但即将恶化的风险，这是去除预测后的明确能力边界；
- 当前没有使用 CLF、MPC、HJ/CBVF 或学习式安全价值函数；
- HOCBF 约束不可行时仍需分级松弛，不能对任意初始状态承诺无碰撞；
- 正式论文还需要多随机种子、置信区间、参数敏感性以及 FSM/v17 枚举、一维/二维、道路约束触发策略等消融实验。
