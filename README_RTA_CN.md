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

## 9. 追尾场景时间窗试验

生成入口新增 `--attack_time`，按固定 0.1 s 回放时钟指定后向接触目标时刻，并要求与模型的 5 帧时间步长对齐。例如 `--attack_time 3.0` 将目标设在第 30 帧；不传该参数时仍使用自车最后有效帧。生成时只截断攻击车的反向生成有效区间，不截断自车参考路线和背景交通。

早于场景末端的接触目标之后，攻击车按目标时刻的速度、航向继续运动，避免在目标处消失。这段轨迹是显式的恒速延续，不属于模型生成，也不是在线追击；该区别记录在 `metadata.rear_attack.post_anchor_motion` 中。时间试验仍以原始记录的自车轨迹设置目标，尚未进行 TD3 轨迹对齐。

```powershell
python bmt/ADV-BMT_rear_attack_dataset_generate.py `
  --dir "Baseline Scenarios/scenarionet_waymo_training_500" `
  --num_scenario 100 --attack_time 3.0 --seed 0 `
  --save_dir "Adv-BMT scenarios/my_rear_attack_3s"

python eval_rear_attack_timing.py `
  --eval_data_dir "Adv-BMT scenarios/my_rear_attack_3s" `
  --ckpt_path "td3_waymo_native/seed_0_300000_steps.zip" `
  --num_episodes 100 --rta_plugin none `
  --out "analysis_outputs/rta_diagnostics/my_rear_attack_3s_none.json"
```

生成脚本和审计入口均拒绝覆盖已有输出。`--start_index` 可用于分段生成；`--seed` 固定一次生成调用的随机种子。时间审计沿用原评估环境及终止配置，额外按 `new_adv_id` 跟踪攻击车，并记录：

- 后车是否实际出现、回合是否覆盖到预设目标帧；
- 是否出现后向接近，以及横向车身投影重叠、后向间隙和接近速度计算得到的 TTC；
- 后向 TTC 不大于 3 s 的威胁是否出现、首次出现在哪一帧；
- 按指定攻击车 ID 进行 Bullet 接触查询的结果及接触方向。

该接触审计与原先 `collision_source` 的最近车辆方向标签分开保存。TTC 是当前相对运动的局部指标，不是轨迹预测，也不等于必然碰撞。RTA 成功使车辆提前横向分离时，后向威胁次数可能减少；因此两组威胁次数不能单独作为攻击场景有效性的筛选条件。

时间锚点回归测试：

```powershell
python -m unittest discover -s tests -p test_rear_attack_timing.py -v
```

2026-11-10 时间窗试验采用同一批 100 个源场景，分别重新生成 9 s 对照和 3 s 提前攻击版本。每组均保留 26 个退化路线场景，并单列 74 个有效动态场景口径；RTA 配置和 TD3 检查点保持一致。

| 指标 | 9 s，无 RTA | 3 s，无 RTA | 3 s，FSM–HOCBF-QP |
|---|---:|---:|---:|
| 有效动态场景中出现后向 TTC≤3 s | 3/74 | 19/74 | 21/74 |
| 覆盖预设接触目标帧 | 0/100 | 40/100 | 49/100 |
| 指定攻击车的后向接触场景数 | 0 | 6 | 4 |
| 全场景碰撞事件率 | 14% | 26% | 9% |
| 全场景到达率 | 65% | 55% | 80% |
| 平均速度 | 11.520 m/s | 10.714 m/s | 9.794 m/s |

3 s 版本中，无 RTA 的后向接触场景编号为 9、10、47、55、75、95；RTA 的后向接触场景编号为 40、65、77、81。后向接触净减少 2 个，但产生了 4 个新的后向接触场景，不能表述为已经解决后向防护问题。RTA 下威胁覆盖次数增加也可能来自回合存活更久或驾驶轨迹变化，不能将其直接当作性能退化指标。

结果位于 `analysis_outputs/rta_diagnostics/timing_trial_comparison_100.json`，完整逐帧审计位于对应的 `timing_*_100.json`。场景位于 `Adv-BMT scenarios/rear_attack_timing_3s_trial_100` 和 `rear_attack_timing_9s_control_100`。这两组是新的开发试验数据，不能与此前 `rear_attack_eval_100` 的结果逐数字混用。生成调用按源索引 0–9、10–96、97–99 分为三段，各段种子均为 0；汇总文件记录了该协议、控制器配置快照和模型哈希。

## 10. 实际后向碰撞专项场景

`mine_rear_collision_scenarios.py` 用固定 TD3 在原生场景中的无 RTA 实际轨迹设置生成参考，输出时仍保留原始自车导航路线和背景交通。接触目标放在有效运行窗口内的 2–3 s，目标中心间距为 3.8 m，目标接近速度增量依次测试 3 m/s、6 m/s。它使用 Adv-BMT 原有 8 模态输出，并优先选择起点确实在后方、方向一致且间距足够的模式。

初始质量检查排除过近起点和明显侧向来车：初始纵向中心距离至少约 8 m、估计车身净间距至少 3 m、横向偏移不超过 3 m、航向差不超过 45°。候选随后由无 RTA 的固定 TD3 闭环运行；仅当指定攻击车发生实际物理接触、接触点位于自车后部、车辆大致同向且 MetaDrive 确认车辆碰撞时才入选。首次接触须在运行至少 1 s 后，每个源场景最多保留一个候选。

专项集的筛选不运行 RTA，也不依赖 TTC。纯危险但未碰撞、前向碰撞、车身中部侧擦、初始重叠和退化路线均不能作为后向碰撞入选依据。`RearAttackAuditEnv` 的 `rear_collision` 使用上述较严格口径，旧的 `rear_attacker_contact` 只检查攻击车中心在后方，两者分别保存。

```powershell
python mine_rear_collision_scenarios.py `
  --source_dir "Baseline Scenarios/scenarionet_waymo_training_500" `
  --ckpt_path "td3_waymo_native/seed_0_300000_steps.zip" `
  --candidate_dir "Adv-BMT scenarios/my_rear_collision_candidates" `
  --selected_dir "Adv-BMT scenarios/my_rear_collision_confirmed" `
  --manifest "analysis_outputs/rta_diagnostics/my_rear_collision_mining.json" `
  --num_sources 150 --target_count 20 --seed 0
```

生成会在候选目录保留所有尝试，并在清单中记录源场景、种子、参数、拒绝原因及实撞审计。`--resume` 可以继续未完成的同一任务；已生成的候选不会覆盖。原有普通追尾场景集独立保留。

这是按“固定无 RTA 基线发生后向碰撞”条件筛选的专项压力测试集。入选集中的高基线碰撞率是筛选条件，不能作为原始交通分布的事故率；应同时报告检查的源场景数、候选数和入选率。该集也不代表已经证明所有场景均存在可行避碰控制。冻结场景后再运行 RTA，两组执行完全相同的固定后车轨迹。

本轮检查 97 个源场景，生成 96 个候选，生成阶段实撞验收通过 20 个。导出后独立复测发现一例未重现碰撞、另有一例未满足严格后部接触判据，因此只按无 RTA 的复测结果保留 18 个。最终场景目录为 `Adv-BMT scenarios/rear_collision_confirmed_v2_18`；该目录再次完整评估后，18/18 均被无 RTA 的 TD3 实际追尾，没有退化路线或仅威胁而未碰撞的样本。场景文件在 RTA 评估前冻结，评估后哈希不变。

| 最终18场景配对结果 | 无 RTA | FSM–HOCBF-QP |
|---|---:|---:|
| 指定攻击车、后部接触点及原生标志确认的后向碰撞 | 18 | 12 |
| 全部车辆碰撞事件 | 18 | 14 |
| 成功到达 | 0 | 2 |
| 道路违规终止 | 0 | 2 |

RTA 的另外两次车辆碰撞未被较严格的后部接触审计确认；它们仍计入全部碰撞，不能当作已避免碰撞。当前控制器在这一高风险专项集中仍有明显不足。摘要为 `analysis_outputs/rta_diagnostics/rear_collision_challenge_v2_summary.json`，完整结果分别为 `rear_collision_confirmed_v2_none_18.json` 和 `rear_collision_confirmed_v2_fsm_18.json`，样本清单及哈希为 `rear_collision_confirmed_v2_18_manifest.json`。原20场景池及开发试验保留用于追溯，不作为最终18场景结果。

## 11. 后向避让开发版（2026-11-11，未替代原默认插件）

新增插件名为 `fsm_rear_escape_cbf_qp`，实现位于 `bmt/runtime_assurance/rear_escape.py`。原 `fsm_hocbf_qp` 和其默认配置保持可用，评估入口的默认插件也没有改动。TD3 检查点、训练代码、奖励函数及 MetaDrive 原生环境未修改。

### 11.1 机动监督与横向分离目标

- 将纵向位置略落后的相邻侧车与真正后方车辆区分，避免侧车宽度主导后向逃逸目标。
- 依据两车的横向车身包络及间距构造横向分离目标，并在一次机动中锁定目标位置；反馈控制使用横向位置误差和横向速度，不再只跟踪固定小航向角。
- 剩余可侧移空间小于 `rear_min_lateral_shift`（0.75 m）时，不把厘米级空隙当作有效机动空间。
- 目标侧移量受当前道路空间限制，保留15%余量。空间不足以单独形成完整横向分离时，可以与纵向控制联合缓解风险，但遥测 `rta_rear_complete_lateral_corridor` 为假，不能将这种状态写成已验证的完整逃逸通道。
- 当前通道检查是几何检查，不是可达性证明；无短时滚动预测、候选动作比较、CLF、MPC或在线学习。

### 11.2 可选后向安全函数及边界

实验分支用当前相对位置和FSM方向构造分离法向量。记自车相对后车的位置为 `p`、速度为 `v_rel`，机动方向为 `m∈{-1,0,1}`，机动开始时锁定的横向单位向量为 `e_lat`，则：

```text
s = p · e_long
w = s² / (s² + bias²)
n = normalize(p + m * bias * w * e_lat)
g = n · p - support_ego(n, heading_ego) - support_rear(n, heading_rear) - margin
h = g - min(n · v_rel, 0)² / (2 * accel_scale)
```

`support` 为包含矩形车身的四阶超椭圆的支撑函数，使用对偶4/3范数计算；`accel_scale` 是速度惩罚的设计尺度，不是已经证明可实现的相对加速度下界。因为减去的是非负项，`h≥0` 蕴含该几何分离条件成立。求导显式计入 `n` 的变化、两车车身航向变化、后车加速度和角速度；结果仍对 `[a, curvature]` 线性。后车角速度在插件中通过航向差分获得并处理角度绕回。

这是**一阶速度型CBF实验约束**，不是原位置安全函数的二阶HOCBF。它替换对应后车的旧约束，不能再叠加旧后向约束后声称消除了旧可行域的限制。模式切换优先保留已经满足 `h≥0` 的约束；若新旧模式均在安全集合外，仅作为恢复控制尝试。`rta_rear_outside_safe_set_count` 及评估汇总会记录安全集合外的运行步数。

保证仍依赖初始安全集、约束可行性、扰动估计界和采样/运动学假设。当前这些假设没有覆盖任意追击后车，也未消除原前侧车/道路HOCBF中的冻结几何近似，因此不能宣称整个系统具有无条件不碰撞保证。

**实测未支持启用新安全函数，因此 `rear_velocity_barrier_enabled` 默认是 `false`。** 当前开发配置仅启用新的机动监督与位置反馈，保留原HOCBF；新函数保留用于后续消融和诊断，不作为性能已提升的正式方法。

### 11.3 同一18场景开发压力集结果

| 方法 | 确认后向碰撞 | 全部碰撞事件 | 成功终止 | 道路违规终止 | 平均奖励 | 完成度 |
|---|---:|---:|---:|---:|---:|---:|
| 无RTA，存档配对基线 | 18 | 18 | 0 | 0 | 23.095 | 0.450 |
| 原FSM–HOCBF，重新复测 | 12 | 14 | 2 | 2 | 37.092 | 0.569 |
| 新机动监督＋新后向CBF，v4 | 14 | 15 | 3 | 0 | 35.915 | 0.544 |
| 新机动监督＋原HOCBF，v5消融 | 11 | 14 | 3 | 1 | 36.050 | 0.553 |

确认后向碰撞采用指定攻击车、后部接触点及原生车辆碰撞标志的联合审计，不是最近车辆方位标签。v5相对旧版新增场景7的成功，但总碰撞数未下降；场景8、13、16的碰撞审计状态也发生变化，不能把确认后向碰撞净少1例直接写成额外避免了1次碰撞。奖励和完成度仍低于旧版。新后向CBF分支没有通过替代原方法的验收。

本轮包含多次方法/参数开发，18场景已成为开发集；结果仅用于诊断，不能包装成独立测试集上的泛化提升。原100场景同时作为回归集，而非重新宣称为独立留出集。

证据文件（均位于 `analysis_outputs/rta_diagnostics/`）：

- `rear_escape_legacy_recheck_18.json`：本轮原FSM复测，与存档主指标一致。
- `rear_escape_v1_challenge_18.json` 至 `rear_escape_v4_challenge_18.json`：各轮实验保留，不覆盖。
- `rear_escape_v5_target_only_18.json`：关闭新后向函数的消融。
- `rear_escape_v5_regression_100.json`：原100场景的开发版回归。

新审计报告保存有效配置、模型哈希、控制器源码哈希及场景哈希，并检查场景文件运行后未改变。较早开发轮次可能没有这些新增字段。计时结果受并行运行和机器负载影响，本轮不据此宣称计算效率改进。

原100场景回归结果（旧版为存档基线，新版为本轮运行）：

| 指标 | 原FSM–HOCBF | 新机动监督＋原HOCBF，v5 |
|---|---:|---:|
| 碰撞事件率 | 4% | 4% |
| 原生越界事件率 | 14% | 14% |
| 到达率 | 87% | 87% |
| 平均奖励 | 40.627 | 40.925 |
| 路线完成度 | 0.681004 | 0.681190 |
| 全步平均速度 | 10.872 m/s | 10.876 m/s |

越界事件率不是互斥的道路终止率；两版互斥终止计数均为成功87、道路违规9、碰撞4。26个退化场景仍包含在100场景口径内，74个有效动态场景数据在原始报告中单列。该回归仅说明现有表现基本持平，奖励小幅变化不能作为显著提升证据。

另一个独立模型检查发现：自车0.5 m/s、后车5.5 m/s、中心间距12 m时，v4参数下左逃逸模式的转向系数约为-0.202；自车10 m/s、后车15 m/s时约为26.086。低速时车身旋转的几何项可能抵消侧向加速度项，故新增函数并不保证始终产生与逃逸方向一致的转向作用。这只是机制边界检查，不能据此断言18个场景的所有失败均由该项造成。此问题与安全集外恢复、约束冲突均未解决，实验函数继续保持关闭。

```powershell
# 开发版：新的横向目标，原HOCBF约束（默认关闭实验安全函数）
python eval_rear_attack_timing.py `
  --eval_data_dir "Adv-BMT scenarios/rear_collision_confirmed_v2_18" `
  --ckpt_path "td3_waymo_native/seed_0_300000_steps.zip" `
  --num_episodes 18 --rta_plugin fsm_rear_escape_cbf_qp `
  --rta_config cfgs/rta_rear_escape_default.json `
  --out "analysis_outputs/rta_diagnostics/my_rear_escape_development.json"

# 控制器单元测试；需在安装了numpy和pytest的Python环境中运行
python -m pytest tests/test_runtime_assurance_cbf2d.py tests/test_runtime_assurance_rear_escape.py -q

# 场景生成/接触审计测试；使用adv-bmt环境
python -m unittest discover -s tests -p "test_rear_*.py" -q
```

## 12. 可避让正向对照与物理一致性修正（2026-11-11）

本轮新增 `diagnose_rta_controlled_rear.py`。它在独立的MetaDrive三车道直路上设置前车和恒速逼近的后车，先验证固定PD避让控制能否实际避撞，再将同一段安全运动作为影子对照，记录哪些CBF约束会排斥其动作。这是离线诊断用的可行运动见证，不是在线动作候选、短时预测或完整可达性分析。

诊断环境使用公开管理器扩展；没有编辑MetaDrive安装目录或修改原生评估环境。不同试验臂禁用对象池复用并使用独立交通车类型，防止诊断用回放车辆的零轮胎摩擦泄漏到下一次自车重置；这一点是本诊断夹具的隔离要求，不据此声称原ScenarioEnv存在同一故障。脚本核对各臂初始位置、速度、车身和动力学参数完全一致，记录每个物理子步的车辆接触，保留原生碰撞和越界终止。

### 12.1 已定位和修正的环节

1. **执行器量纲与轴距。** 旧插件将半制动映射为-3 m/s²；DefaultVehicle独立实测约-7.97 m/s²。新模式读取当前车辆质量、发动机力、制动参数和物理子步，使用每轮驱动力/制动冲量的名义映射，并读取前后轴距之和。该例新半制动映射约-8.13 m/s²，已显著接近实测，但仍不是精确轮胎/车辆动力学。
2. **实际可用输入。** 最大纵向加速度不超过当前车辆名义驱动力；曲率上界同时受几何转向极限和 `a_lat_max/max(v²,1)` 限制，避免QP请求高速大曲率。无威胁直通也不能绕过启用模式的输入范围。
3. **纵横向跟踪代价。** 新模式比较纵向加速度与横向加速度的偏差，而不是固定权重下直接混合加速度与曲率：`J = W_a*(a-a_ref)^2 + W_lat*v^4*(curvature-curvature_ref)^2`（极低速使用1 m/s的数值下限）。不可行回退在 `z=sqrt(W)*u` 坐标中预条件化，保持同一优化目标、约束和软惩罚，避免病态权重导致有限迭代下的求解偏差。
4. **车身角点及侧向速度。** 新模式的道路约束使用 `h_corner = geometric_gap + tau(v)*smooth_min(inward_velocity,0)`，其中 `tau=L/sqrt(v²+0.1²)+0.2`。安全函数不大于真实角点几何间隙；求导保留车身旋转、侧向加速度与速度相关系数的变化，仍对加速度和曲率线性。新模式持续执行这些道路约束，不能因当前距离尚大而让避车修正绕开道路约束。
5. **状态连续性。** 速度相关输入界和权重每周期更新，但不重建FSM，不丢失驻留计时和已锁定的逃逸目标。

上述变化仅通过 `cfgs/rta_rear_physical.json` 启用。旧 `fsm_hocbf_qp` 保持原有行为；第11节未通过验证的实验后向支撑面函数仍关闭。

### 12.2 后向专用监管插件

新增 `fsm_rear_physical_guard`：无确认后向逼近时使用原FSM–HOCBF；出现当前状态后向风险且道路安全函数处于允许的初始化/恢复范围时，切入物理一致模式。当前恢复范围为 `h_road >= -0.25`，负值进入时仅是恢复尝试，不能按安全集合内前向不变性表述。后车通过或已分离、横向运动稳定，且原控制器的名义动作满足其当前约束后，才退回原模式。

监管器只根据当前状态判据选择模式，每周期仅执行被选中的控制器；不求解两套候选动作后比较，也不读取场景攻击车标签作控制决策。该结构用于避免全局套用开发中的动力学/道路约束而损害其他场景。模型和CBF可行性仍有边界，不提供任意攻击下的无条件保证。

### 12.3 可重复正向对照

自车初速分别0.5、3、10 m/s，后车均快5 m/s，初始中心距离12 m，前车在14 m前方并保持自车初始速度，固定运行6 s：

| 自车初速 | 无RTA名义控制 | 原FSM–HOCBF | 后向监管修正版 |
|---|---|---|---|
| 0.5 m/s | 被后车撞击 | 被后车撞击 | 无碰撞、无越界，完成6 s |
| 3 m/s | 被后车撞击 | 越界终止 | 无碰撞、无越界，完成6 s |
| 10 m/s | 被后车撞击 | 越界终止 | 无碰撞、无越界，完成6 s |

三个场景均独立验证了PD见证无碰撞、无越界。原/开发CBF在见证轨迹上仍会排斥部分动作，这证明这些安全条件没有覆盖全部可行避让运动，不等于证明CBF的一般理论错误。修正版在这些例子中也仍出现硬约束不可行和松弛，不能把物理避撞成功等同于已经完成形式化安全证明。

### 12.4 冻结数据开发复测

同一18场景、同一TD3检查点，原FSM为本轮已复测基线，后向监管采用v2：

| 指标 | 原FSM–HOCBF | 后向监管v2 |
|---|---:|---:|
| 严格确认后向碰撞 | 12 | 10 |
| 全部碰撞事件 | 14 | 13 |
| 成功到达 | 2 | 4 |
| 道路违规终止 | 2 | 1 |
| 平均奖励 | 37.092 | 41.524 |
| 路线完成度 | 0.56896 | 0.59694 |

严格后向接触少2例不代表完整避免了2次碰撞；全部碰撞净少1例。回合长度和结局改变，不能用专项集的全步均速变化单独宣称效率提升。

原100场景回归（旧版为存档基线）：

| 指标 | 原FSM–HOCBF | 后向监管v2 |
|---|---:|---:|
| 碰撞事件率 | 4% | 4% |
| 原生越界事件率 | 14% | 13% |
| 到达率 | 87% | 88% |
| 平均奖励 | 40.627 | 40.890 |
| 路线完成度 | 0.68100 | 0.68578 |
| 全步平均速度 | 10.872 m/s | 10.892 m/s |

这些是小样本、多轮开发后的初步改善，尚不构成独立留出集上的显著性或泛化结论。18个场景仍有13例碰撞，原100场景仍有26个退化场景。本轮不自动替换评估入口默认插件；原有结果和各轮失败实验均保留。

主要证据位于 `analysis_outputs/rta_diagnostics/`：

- `rear_guard_v2_challenge_18.json`、`rear_guard_v2_regression_100.json`：本轮监管版结果。
- `controlled_rear_guard_v1_*mps.json`及原生集成测试生成的报告：可避让正向对照。
- `controlled_rear_fixture_validated_10mps.json`：修正前对照及执行器测量。
- `rear_physical_v3_*`、`rear_physical_v4_*`、`rear_guard_v1_*`：全局启用及监管过程的开发记录，不应只选择有利指标报告。

```powershell
# 后向监管插件，按需启用；原默认插件不变
python eval_rear_attack_timing.py `
  --eval_data_dir "Adv-BMT scenarios/rear_collision_confirmed_v2_18" `
  --ckpt_path "td3_waymo_native/seed_0_300000_steps.zip" `
  --num_episodes 18 --rta_plugin fsm_rear_physical_guard `
  --rta_config cfgs/rta_rear_physical.json `
  --out "analysis_outputs/rta_diagnostics/my_rear_guard_run.json"

# 可避让正向对照；不通过时保存报告并返回非零退出码
python diagnose_rta_controlled_rear.py --speed 3 --rear_speed 8 `
  --plugin fsm_rear_physical_guard --rta_config cfgs/rta_rear_physical.json `
  --require_rta_safe --out "analysis_outputs/rta_diagnostics/my_controlled_case.json"

# 普通单元测试（不要求当前Python安装MetaDrive）
python -m pytest tests/test_runtime_assurance_cbf2d.py `
  tests/test_runtime_assurance_rear_escape.py tests/test_runtime_assurance_physical.py -q

# 当前启用3种速度×4种配置的真实物理测试；报告使用唯一文件名，拒绝覆盖
$env:RTA_NATIVE_TEST_PYTHON = "D:\SpecializedSoftware\Anaconda\envs\adv-bmt\python.exe"
python -m pytest tests/test_runtime_assurance_controlled_integration.py -q
```

## 13. 道路几何与机动持续性调试：候选版，尚未达到目标

本轮目标是：在旧 FSM 发生碰撞的 14 个固定场景中，至少使 10 个无碰撞、无越界并到达终点。**当前只达到 4/14，不应写成目标已实现，也不将候选版替换为默认配置。** 模型、场景、碰撞终止设置和评估时域均未改变；所有试验继续使用相同的 TD3 300k 检查点。

### 13.1 定位及修改

- ScenarioEnv 的 `TrajectoryNavigation` 使用默认宽度为 2 m 的虚拟参考轨迹。其 `dist_to_left_side/right_side` 在轨迹中心约为 1/3 m，不能解释为真实道路边界。新配置只在该导航类型下改用原生连续线/路缘探测距离；有效无命中表示传感器测程内无命中，传感器失效则保留旧约束，不直接取消道路约束。
- 程序化地图导航缺少轨迹航向属性时，旧适配器使用自车航向作为道路航向。新配置读取原生地图车道切向，防止避让时道路坐标系随车头旋转。该问题由 0.5 m/s 原生物理正向对照的新增失败暴露，修复后重新验证。
- 逃逸空间评估可围绕已锁定目标检查剩余位移，不再每步要求重新完成至少 0.75 m 的新横移；同时保留对实际占据侧方通道车辆的检查。仍在后方、尚未进入横移扫掠区的车辆不作为目标位置的静态占据，但其车辆安全约束继续进入 QP。
- 使用当前净间距与闭合速度计算 TTC 作为补充风险触发量；不做动作枚举或轨迹滚动预测。选择方向时区分能实现车体分离的通道和只能部分横移的通道。
- 对后车脱离后的回归参考轨迹及后向车辆航向一致性进行了独立试验。这两项在当前专项集没有额外增加安全完成数，不作为已验证的性能贡献。

以上功能均为显式配置开关。旧 `fsm_hocbf_qp` 和 `cfgs/rta_rear_physical.json` 的默认开关保持关闭。训练、奖励函数和 MetaDrive 安装目录未修改。

### 13.2 固定 18 场景配对结果

| 指标 | 旧 FSM | 上轮物理监管版 | 本轮道路几何候选 |
| --- | ---: | ---: | ---: |
| 全部碰撞数 | 14 | 13 | 8 |
| 严格确认后向碰撞数 | 12 | 10 | 6 |
| 无碰撞、无越界到达数 | 2 | 4 | 7 |
| 原生越界事件数 | 2 | 1 | 3 |
| 平均奖励 | 37.092 | 41.524 | 42.555 |
| 平均路线完成度 | 0.5690 | 0.5969 | 0.5916 |

注意：相比上轮，平均路线完成度没有提升。也不能只展示碰撞下降而忽略越界增加。

原 14 个碰撞场景的固定队列为 `0,1,2,4,5,6,8,9,12,13,14,15,16,17`。本轮其中：

- `8,12,13,14`：无碰撞、无越界并到达，4 个；
- `0,4`：未碰撞但越界，2 个，不计成功；
- `1,2,5,6,9,15,16,17`：仍碰撞，8 个。

因此“14 次碰撞降为 8 次”不等于“安全解决了 6 个场景”，严格安全完成的改善是 4 个，尚未达到 10 个。剩余样本没有全部获得物理可避让证据，不能把 QP 不可行解释为碰撞必然不可避免。

### 13.3 普通 100 场景回归

| 指标 | 上轮物理监管版 | 本轮候选 |
| --- | ---: | ---: |
| 全部碰撞率 | 4% | 3% |
| 到达率 | 88% | 86% |
| 越界终止数 | 8 | 8 |
| 超时或其他终止数 | 0 | 3 |
| 原生越界事件率 | 13% | 13% |
| 平均奖励 | 40.890 | 39.685 |
| 平均路线完成度 | 0.6858 | 0.6757 |
| 平均速度（m/s） | 10.892 | 10.519 |

同一无 RTA 档案的平均速度为 11.608 m/s，候选版降幅约 9.38%，未满足此前约 7% 的效率期望。因此该配置仅作为调试候选保留，不认定为整体优于上轮。100 场景仍包含同样的 26 个退化场景、74 个有效动态场景，未根据新结果改变分母。18 场景已经用于多轮开发，不能当作未见独立测试集。

### 13.4 开发记录与复现

候选配置为 `cfgs/rta_rear_roadfix_candidate.json`，与 `cfgs/rta_rear_physical_v16_aligned.json` 内容相同。这里的 `rear_physical_v3` 至 `v16` 是本轮物理监管配置的开发编号，与早期枚举控制器版本号无关。

保留了失败尝试：全局启用、实验后向支持函数、边界法向拟合、角点探测、后向 HOCBF 增益修改、降低软约束惩罚、缩短横向目标及扩大原生转向范围。部分版本把碰撞降至 6 次，但越界增至 5～6 次，仍只有 4 个原碰撞场景安全完成，未晋级。它们不是在线枚举控制候选，而是离线开发试验。

- 完整配对摘要与所有开发版本指标：`analysis_outputs/rta_diagnostics/rear_roadfix_debug_summary.json`。
- 候选专项结果：`rear_guard_v16_aligned_18.json`；普通回归：`rear_guard_v16_aligned_100.json`。
- 以上两份候选报告的控制器源码哈希相同，并与本轮归档时源码一致；检查点及 18 场景哈希与旧配对基线相同，场景文件哈希未改变。
- `diagnose_rear_reference.py` 仅用于离线执行未经过 QP 的参考动作以定位约束影响；输出明确标为 `not_rta`，不得作为安全方法的结果。
- 最后验证：121 项 pytest（含 12 项原生物理集成测试）及 9 项场景生成/筛选 unittest 通过，共 130 项；`git diff --check` 通过。场景集、检查点及候选控制器源码哈希检查通过。

```powershell
python eval_rear_attack_timing.py `
  --eval_data_dir "Adv-BMT scenarios/rear_collision_confirmed_v2_18" `
  --ckpt_path "td3_waymo_native/seed_0_300000_steps.zip" `
  --num_episodes 18 --rta_plugin fsm_rear_physical_guard `
  --rta_config cfgs/rta_rear_roadfix_candidate.json `
  --out "analysis_outputs/rta_diagnostics/my_roadfix_candidate_18.json"
```

目前仍启用原来的后向车辆 HOCBF，实验替代函数没有晋级。修正道路和监督逻辑后仍存在大量后向约束不可行、参考避让方向与实际 QP 控制不一致的步骤；这说明不能只依赖继续增大转向或放松惩罚来实现本轮目标。下一阶段需要针对可避让失败样本重新设计机动条件化的后向约束，并验证前车、侧车和道路次生风险，而不是把越界或未被严格归类为后向碰撞当成成功。

## 14. 速度降幅 5% 目标试验（2026-11-13，未晋级）

本轮尝试降低第 13 节候选的速度损失。仍使用原 100 个场景、同一 TD3 检查点和原有 `avg_speed_mps` 全步均速口径，不筛掉慢场景、不改成共同成功场景、不重新加权；同时检查固定 18 个后向碰撞场景及 0.5、3、10 m/s 的原生物理正向对照。仅修改独立 RTA 插件及其开发配置、测试和离线摘要工具，未修改策略、训练、奖励、原生 MetaDrive、场景文件或原有默认配置。

### 14.1 本轮基线与速度目标

重新运行无 RTA 的 100 场景得到均速 **11.601466 m/s**；5% 降幅要求至少 **11.021393 m/s**，7% 要求至少 **10.789363 m/s**。原候选为 **10.519098 m/s**，相对本轮基线降幅 **9.33%**。

本轮带接触审计与不带审计的无 RTA 入口得到相同均速、碰撞数和终止结果：碰撞事件 17、到达 64、越界事件 29。此前存档基线为 11.607906 m/s、碰撞事件 14；速度相差约 0.055%，但碰撞数存在差异，原因尚未确定，不能混用这两轮碰撞率，也不能将差异归因于审计。以下速度降幅统一使用本轮重新运行的 11.601466 m/s。原候选也已在本轮源码下重新执行 100/18 两组完整评估，其均速、奖励、完成度、碰撞率、越界率及到达率均与原存档完全一致。

### 14.2 定位结果及试验边界

- **接管返回时的道路约束不一致。** 旧逻辑用物理道路检查是否可以返回，却交给使用虚拟参考轨迹宽度的原控制器，可能返回后立即触发道路停止。新增 `rear_guard_consistent_handoff`，按实际接收控制器的道路及障碍物约束检查返回条件。该修正单独带来的均速提升很小，不能解释全部速度损失。
- **纵向上限与动作代价。** 将配置加速度上限从 3 改为 4，实际控制仍受当前车辆原生驱动力上限限制，不增加车辆动力。提高纵向跟踪权重至 4 虽能改善部分均速，却在已知可避让的 0.5 m/s 正向对照中重新发生后向碰撞，因此相关方案被拒绝。
- **低紧迫性后车触发完整横移。** 新增基于当前 TTC、`h`、`psi1` 的可选接管筛选，未接管时仍执行原车辆和道路 CBF，不绕过 RTA。但延后机动也可能错过避让时机；专项集确实出现后向碰撞增加，不能只按均速选择配置。
- **机动跟踪和解除。** 试验了瞬时纵向输入可行区间、保持已启动机动、道路局部坐标横移目标、后车脱离后释放以及低速道路域纠正。这些均为显式关闭的开发开关，不是已验证的性能贡献，也不涉及轨迹滚动预测或在线动作枚举。瞬时约束可行不等于完整可达性保证。

诊断中特别保留了：场景 99 接管返回后的虚拟道路停止、场景 36 的低速约束冲突，以及场景 88 大转角且道路航向不一致时低速持续的问题。并非所有减速都来自不必要制动；部分来自横向运动、前车约束和原候选的超时。

### 14.3 已拒绝的主要试验

以下为开发试验，不是在线多候选控制，也不是独立未见测试集结果。速度来自完整 100 场景；专项列为 18 场景。事件标志可能重叠，不要求碰撞、越界和原始到达三项相加等于 100%。

表格记录各试验当时的源码行为，配置编号不是完整源码快照。尤其 v10/v11 的宽泛接管随后已在源码中限制为低速；用当前源码运行这两个同名配置，不保证重现原表结果。报告保存的源码哈希只能识别差异，不等于保存了可恢复的每轮源码。它们仅作为调试记录，正式实验需另行冻结完整源码与配置。

| 配置 | 100 场景均速 m/s | 降幅 | 碰撞事件 | 越界事件 | 到达 | 专项碰撞 / 严格后向碰撞 / 干净到达 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原候选 v16 复测 | 10.5191 | 9.33% | 3 | 13 | 86 | 8 / 6 / 7 |
| v1 一致返回检查 | 10.5234 | 9.29% | 3 | 13 | 86 | 未运行专项 |
| v2 纵向权重 4 | 10.6632 | 8.09% | 4 | 14 | 86 | 7 / 5 / 7 |
| v5 原生驱动力上限及接管筛选 | 10.8113 | 6.81% | 3 | 14 | 87 | 8 / 7 / 7 |
| v6 瞬时纵向可行性 | 10.7477 | 7.36% | 3 | 13 | 87 | 8 / 6 / 5 |
| v7 保持已启动机动 | 10.7466 | 7.37% | 3 | 13 | 87 | 8 / 6 / 6 |
| v8 局部道路坐标跟踪 | 10.4865 | 9.61% | 3 | 13 | 86 | 8 / 6 / 6 |
| v9 后车脱离后解除 | 10.7249 | 7.56% | 3 | 13 | 87 | 9 / 8 / 6 |
| v10 宽泛道路域纠正 | 10.6714 | 8.02% | 3 | 17 | 83 | 8 / 7 / 7 |
| v12 仅低速道路域纠正 | 10.7073 | 7.71% | 3 | 13 | 87 | 8 / 7 / 7 |
| v13 保留纵向权重 1 | 10.7072 | 7.71% | 3 | 13 | 87 | 9 / 8 / 7 |

**v5 的约 6.8% 不能作为优化成功结论。** 它比原候选多一个越界事件、专项多一个严格确认后向碰撞，且在原生 0.5 m/s 可避让正向对照中重新追尾。场景 36 从超时变为早期越界，回合长度改变也影响全步均速，因此不能把均速提高全部解释为行驶更高效。v12 同样未通过该低速正向对照。

v13 保留原纵向权重 1，仅采用一致返回检查、原生驱动力上限和接管筛选；专项结果为全部碰撞 **9**、严格后向碰撞 **8**、干净到达 **7**、越界事件 **2**。其三个原生物理正向对照均无碰撞、无越界并完成 6 s，但专项全部碰撞多于原候选的 8 次，故同样不晋级。v3、v4 仅创建配置但未评估；v11 为宽泛道路域纠正的权重 1 变体，普通回归越界事件仍为 17，未运行专项。

### 14.4 配置、结果与复现

`cfgs/rta_efficiency_v*.json` 全部为实验配置；**不自动替换** `cfgs/rta_rear_roadfix_candidate.json`、`cfgs/rta_rear_physical.json` 或评估入口默认值。新增开关的类型默认值均为关闭。

- `analysis_outputs/rta_diagnostics/speed5_*_100.json`、`speed5_*_18.json`：各次固定场景结果及模型、场景、控制器源码哈希。
- `speed5_none_refresh_100.json`、`speed5_none_uninstrumented_100.json`：本轮无 RTA 基线与无审计入口复核。
- `speed5_before_refresh_100.json`、`speed5_before_refresh_18.json`：原候选同批复测；`speed5_summary_final.json` 为以本轮复测作参照的最终摘要。
- `native_integration_*.json`：真实物理正向对照，包含初始条件、可行避让见证、实际接触和开发配置。v5/v12 的 0.5 m/s 已知失败在测试中显式标为严格 `xfail`，原安全断言仍执行；不能把这些当成通过或部署配置。
- `summarize_rta_efficiency.py`：核对完整配对场景、检查点、控制步长、退化场景数及两个数据集的安全指标，拒绝覆盖输出。摘要中的固定队列安全门槛不能代替原生正向对照，也不会自动晋级配置。

本轮单元测试 122 项、场景生成/筛选 unittest 9 项通过。另逐份核对本轮 21 份原生物理报告：19 个对照通过，2 个为明确拒绝的 v5/v12 低速后向碰撞，不能记作安全通过。

最终 v13 的两组评估具有相同控制器源码哈希，并与当前 RTA 源码一致；同一 TD3 检查点哈希及配对场景文件检查通过。100 场景仍包含原有 26 个退化场景；`git diff --check` 通过。本轮没有自动提交 Git。

```powershell
# 只读汇总已有完整报告；使用尚不存在的新输出文件名
python summarize_rta_efficiency.py `
  --before_100 speed5_before_refresh_100.json --before_18 speed5_before_refresh_18.json `
  --out analysis_outputs/rta_diagnostics/my_speed_summary.json

# 单元测试
python -m pytest tests/test_runtime_assurance_cbf2d.py `
  tests/test_runtime_assurance_rear_escape.py tests/test_runtime_assurance_physical.py `
  tests/test_runtime_assurance_efficiency_summary.py -q

# 含严格安全断言的原生物理回归；实验失败会单独显示
$env:RTA_NATIVE_TEST_PYTHON = "D:\SpecializedSoftware\Anaconda\envs\adv-bmt\python.exe"
python -m pytest tests/test_runtime_assurance_controlled_integration.py -q
```

当前结论：**尚未得到兼顾原候选安全结果且将速度损失降至 5% 的新配置；约 6.8% 的试验方案也因安全退化被拒绝。** 两个固定数据集均已用于调试，不能将多轮试参后的最好结果当作留出集结论。
