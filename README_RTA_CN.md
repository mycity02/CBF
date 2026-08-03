# Adv-BMT 二维 HOCBF-QP 运行时保障扩展

本版本在原始 Adv-BMT 项目上增加了一个面向 TD3 驾驶策略的二维运行时保障（Runtime Assurance，RTA）插件。插件用于 Adv-BMT 追尾场景的评估阶段，在不重新训练 TD3、不修改奖励函数、不改变策略观测且不修改 MetaDrive 原生环境代码的前提下，对 TD3 输出的转向和加减速动作进行实时安全过滤。

> 当前版本是研究原型。它采用三组方向性参考控制与短时运动学前视，不是严格的三模式混合控制系统，也尚未提供对 MetaDrive 完整动力学下全局安全性的形式化保证。

## 1. 主要功能

- 同时处理前车、后车和侧向交通参与者带来的碰撞风险；
- 将道路左右边界加入安全约束，降低避碰过程中驶出道路的概率；
- 使用二维高阶控制障碍函数（HOCBF）约束纵向加速度与路径曲率；
- 使用无额外优化器依赖的二维凸 QP 计算连续安全控制；
- 保留 TD3 原始控制，并额外生成左偏、右偏两组参考控制；
- 通过短时运动学前视比较候选控制的未来安全裕度；
- 在硬约束不可同时满足时采用分级回退，并记录约束可行性；
- 输出干预率、障碍函数值、候选选择、速度和计算延迟等安全遥测数据；
- RTA 入口与原始评估入口相互隔离，可通过参数启用或关闭插件。

## 2. 系统位置

RTA 只作用于策略执行阶段：

```text
场景观测
   ↓
TD3.predict()
   ↓
原始动作 [steering, throttle/brake]
   ↓
predictive_escape_cbf2d 插件
   ↓
安全动作 [steering, throttle/brake]
   ↓
MetaDrive env.step()
```

原始评估脚本 `eval_td3_policy.py` 保持不变；带 RTA 的独立入口为 `eval_td3_policy_rta.py`。插件不会修改 TD3 的参数、输入观测和奖励，也不会参与训练。

## 3. 方法说明

### 3.1 控制变量与运动学模型

MetaDrive 适配器首先将归一化策略动作转换为物理控制量：

\[
u=[a,\kappa]^\mathsf{T},
\]

其中，\(a\) 为纵向加速度，\(\kappa\) 为路径曲率。RTA 内部使用曲率参数化的二维运动学近似：

\[
\dot x=v\cos\theta,\qquad
\dot y=v\sin\theta,
\]

\[
\dot v=a,\qquad
\dot\theta=v\kappa.
\]

周围交通参与者在短时前视中采用世界坐标系下的恒加速度模型：

\[
p_{k+1}=p_k+v_k\Delta t+\frac{1}{2}a_k\Delta t^2,
\qquad
v_{k+1}=v_k+a_k\Delta t.
\]

上述模型只用于在线安全过滤和候选评估，最终车辆状态仍由 MetaDrive 原生仿真动力学推进。

### 3.2 二维 HOCBF 约束

车辆外形使用局部道路坐标系下的平滑超椭圆进行保守包络。车辆避碰约束与道路边界约束被转换为关于 \(a\) 和 \(\kappa\) 的二阶 HOCBF 线性不等式。在线 QP 在满足物理控制范围和当前激活的安全约束时，尽量减小安全控制与参考控制之间的差异：

\[
u_m^*=\arg\min_u\|u-u_m^{\mathrm{ref}}\|_W^2.
\]

这里使用的是硬安全约束优先的分级投影。若车辆约束和道路约束不存在公共解，则道路约束在物理可行时保持为强制约束，车辆约束按照归一化违反量进行松弛，并将回退状态写入遥测。

### 3.3 三组方向性参考控制

TD3 给出的物理控制记为：

\[
u_{\mathrm{TD3}}=(a_{\mathrm{TD3}},\kappa_{\mathrm{TD3}}).
\]

系统以此为基础构造三组参考控制：

- `nominal`：保留 TD3 原始加速度和曲率；
- `left_escape`：保留 TD3 加速度，增加向左的曲率倾向；
- `right_escape`：保留 TD3 加速度，增加向右的曲率倾向。

左右参考曲率会根据车辆速度、前视时间和当前道路可用宽度进行调整。三组参考分别投影到**同一组** HOCBF 可行控制集，输出的加速度和曲率仍然是连续量。

因此，这里的三个候选不是三个固定动作，也不是具有不同约束集合 \(A_m(x)u\le b_m(x)\) 的严格控制模式。准确表述应为：

> 有限方向参考生成—HOCBF-QP 连续安全修正—短时运动学前视选择。

### 3.4 短时前视与候选选择

QP 输出的三组连续控制在短时间内按恒定控制进行运动学前视。评分主要考虑：

- 车辆和道路约束的未来最小安全裕度；
- 相对 TD3 原始控制的动作偏离代价；
- QP 是否硬可行以及归一化约束违反程度。

系统选择评分最高的候选作为当前控制周期的安全动作。该机制用于打破正后方威胁下的左右对称性，但其候选覆盖性和参数敏感性仍需要进一步研究。

## 4. 目录与入口

```text
eval_td3_policy.py                         原始 TD3 评估入口，不启用 RTA
eval_td3_policy_rta.py                     可插拔 RTA 评估入口
diagnose_rta_scenario.py                   单场景逐步诊断脚本
bmt/runtime_assurance/
├── adaptive_plugin.py                     自适应方向参考与公开插件入口
├── cbf2d.py                               二维 HOCBF、候选前视与选择
├── metadrive_plugin.py                    MetaDrive 状态/动作适配器
├── qp.py                                  二维优先级 QP 求解器
├── metrics.py                             速度与运行延迟统计
└── types.py                               配置、状态和遥测数据结构
cfgs/
├── rta_cbf2d_default.json                 基础配置
├── rta_cbf2d_side_vru_completion.json     v15 完成度配置
├── rta_cbf2d_vehicle_hard_road_preview.json  v17 奖励配置
└── rta_cbf2d_side_detector_no_vru.json    不含弱势道路使用者的消融配置
tests/
├── test_runtime_assurance_cbf2d.py        HOCBF、QP 与插件测试
└── test_runtime_assurance_metrics.py      速度和延迟指标测试
analysis_outputs/rta_diagnostics/          已有评估结果与诊断轨迹
```

其中 `v15` 和 `v17` 是本项目调参过程中的实验标签，不是对外发布的软件语义版本号。

## 5. 环境准备

RTA 复用原始 Adv-BMT、MetaDrive 和 TD3 评估环境。建议先按照项目根目录的 `readme.md` 完成环境安装，并确认原始 `eval_td3_policy.py` 能够正常运行，再启用 RTA。

仓库包含本地 MetaDrive 和 ScenarioNet 源码时，可以在项目目录执行：

```powershell
conda create -n adv-bmt-rta python=3.10 -y
conda activate adv-bmt-rta

python -m pip install -r requirements.txt
python -m pip install -e .\metadrive
python -m pip install -e .\scenarionet
python -m pip install -e .
```

TD3 模型加载还依赖与原训练环境一致的 Stable-Baselines3/自定义 TD3 实现。若已有能够运行原始 TD3 评估脚本的环境，建议直接复用，避免因依赖版本变化导致旧检查点无法加载。

可使用以下命令检查关键导入：

```powershell
python -c "from metadrive.envs.scenario_env import ScenarioEnv; from bmt.rl_train.train.customized_td3 import Closed_Loop_TD3; print('环境导入正常')"
```

## 6. 快速运行

下面的命令使用仓库现有的原生场景训练 TD3 检查点和 100 个 Adv-BMT 追尾评估场景。实验对比时必须保持检查点、场景目录、起始索引、回合数、决策时域和随机性设置一致。

### 6.1 无 RTA 基线

```powershell
python eval_td3_policy_rta.py `
  --ckpt_path "td3_waymo_native\seed_0_300000_steps.zip" `
  --eval_data_dir "Adv-BMT scenarios\rear_attack_eval_100" `
  --num_episodes 100 `
  --horizon 100 `
  --start_index 0 `
  --device cpu `
  --rta_plugin none `
  --out "analysis_outputs\rta_diagnostics\my_no_rta.json"
```

该命令通过同一个扩展评估入口关闭插件，便于获得完全一致的速度、时延和配对指标。也可以使用未修改的原始入口进行独立核验：

```powershell
python eval_td3_policy.py `
  --ckpt_path "td3_waymo_native\seed_0_300000_steps.zip" `
  --eval_data_dir "Adv-BMT scenarios\rear_attack_eval_100" `
  --num_episodes 100 `
  --horizon 100 `
  --start_index 0 `
  --device cpu `
  --out "analysis_outputs\native_eval.json"
```

### 6.2 v15：完成度优先配置

```powershell
python eval_td3_policy_rta.py `
  --ckpt_path "td3_waymo_native\seed_0_300000_steps.zip" `
  --eval_data_dir "Adv-BMT scenarios\rear_attack_eval_100" `
  --num_episodes 100 `
  --horizon 100 `
  --start_index 0 `
  --device cpu `
  --rta_plugin predictive_escape_cbf2d `
  --rta_config "cfgs\rta_cbf2d_side_vru_completion.json" `
  --out "analysis_outputs\rta_diagnostics\my_rta_v15.json"
```

### 6.3 v17：奖励优先配置

```powershell
python eval_td3_policy_rta.py `
  --ckpt_path "td3_waymo_native\seed_0_300000_steps.zip" `
  --eval_data_dir "Adv-BMT scenarios\rear_attack_eval_100" `
  --num_episodes 100 `
  --horizon 100 `
  --start_index 0 `
  --device cpu `
  --rta_plugin predictive_escape_cbf2d `
  --rta_config "cfgs\rta_cbf2d_vehicle_hard_road_preview.json" `
  --out "analysis_outputs\rta_diagnostics\my_rta_v17.json"
```

如需随机策略动作，可额外添加 `--stochastic`；论文式配对比较建议保持默认的确定性推理。

## 7. 单场景诊断

出现碰撞、越界、低完成度或不可行回退时，可记录指定场景的逐步诊断轨迹：

```powershell
python diagnose_rta_scenario.py `
  --ckpt_path "td3_waymo_native\seed_0_300000_steps.zip" `
  --eval_data_dir "Adv-BMT scenarios\rear_attack_eval_100" `
  --scenario_index 0 `
  --horizon 100 `
  --device cpu `
  --rta_plugin predictive_escape_cbf2d `
  --rta_config "cfgs\rta_cbf2d_side_vru_completion.json" `
  --out "analysis_outputs\rta_diagnostics\scenario_0_trace.json"
```

诊断文件包含每步原始动作、安全动作、选择的方向参考、障碍函数值、QP 可行性、约束违反量、道路几何来源和候选评分。

## 8. 当前实验结果

以下结果来自同一个原生场景训练 TD3 检查点、相同的 100 个 Adv-BMT 追尾场景、确定性推理、100 步时域和 CPU 执行。数值来自 `analysis_outputs/rta_diagnostics/paired_100_speed_comparison.md`。

| 方法 | 平均奖励 | 平均完成度 | 碰撞率 | 越界率 | 到达率 | 全步骤平均速度 |
|---|---:|---:|---:|---:|---:|---:|
| 无 RTA | 29.759 | 0.677 | 14% | 37% | 72% | 11.386 m/s |
| v15 完成度配置 | 32.228 | 0.702 | 6% | 30% | 76% | 10.329 m/s |
| v17 奖励配置 | 33.167 | 0.693 | 5% | 33% | 75% | 10.446 m/s |

在无 RTA 与相应 RTA 版本均到达终点的配对场景中：

- v15 的平均速度下降约 2.0%，平均用时增加约 3.2%；
- v17 的平均速度下降约 1.9%，平均用时增加约 3.5%。

全步骤平均速度会受到终止偏差影响：RTA 可能让原本因越界而提前终止的困难场景继续运行，因此不应只根据该指标评价效率。

### RTA 计算延迟

MetaDrive 当前控制周期为 100 ms，延迟只统计 `filter_action()`，不包含 TD3 推理和 MetaDrive 物理仿真。

| 配置 | 平均延迟 | P95 | 超过 100 ms 的比例 |
|---|---:|---:|---:|
| v15 | 16.04 ms | 44.88 ms | 1.87% |
| v17 | 19.83 ms | 67.60 ms | 3.38% |

v15 的实时性更强，v17 的平均奖励和碰撞率更好，但尾部延迟略高。

这些结果属于当前固定检查点与固定场景集上的初步实验，不等同于跨场景泛化结论或形式化安全证明。正式论文实验还应补充多随机种子、置信区间或显著性检验，以及一维/二维、无前视、不同候选机制和模型误差鲁棒性等消融实验。

## 9. 输出指标与遥测

评估 JSON 的 `summary` 提供：

- `reward`：平均回合奖励；
- `completion`：平均路线完成度；
- `collision`：发生碰撞的回合比例；
- `out_of_road_rate`：越界回合比例；
- `arrive_dest_rate`：到达终点的回合比例；
- `avg_speed_mps`：所有控制步的平均速度；
- `rta_intervention_rate`：RTA 修改原始动作的控制步比例；
- `rta_avg_infeasible_steps`：每回合平均硬不可行步数；
- `rta_filter_mean_ms/p95_ms/p99_ms`：过滤器延迟；
- `rta_deadline_miss_rate`：超过当前控制周期的调用比例。

每个回合还记录碰撞方向近似分类、模式计数、最小车辆/道路障碍函数值、行驶距离、回合时间和到达场景速度。插件逐步遥测进一步提供：

- 原始与安全加速度、曲率和 MetaDrive 动作；
- `nominal/left_escape/right_escape` 候选评分；
- 激活障碍物数量；
- HOCBF 的 \(h\)、\(\dot h\)、\(\psi_1\) 和残差；
- 硬可行性、强制道路约束可行性和回退类型。

## 10. 测试

RTA 单元测试不需要运行完整的 100 场景评估：

```powershell
python -m pytest tests\test_runtime_assurance_cbf2d.py tests\test_runtime_assurance_metrics.py -q
```

测试覆盖二维 QP 投影、最小干预、预测激活、前后夹击时的横向逃逸、侧车方向选择、道路约束优先回退、道路侧向探测器适配、速度统计和实时期限统计。

## 11. 当前局限与后续方向

- 三组方向性参考是人工设定的有限候选，并非严格的混合 HOCBF 模式；
- 在线前视采用局部运动学模型和恒定控制，未覆盖完整 MetaDrive 动力学误差；
- 周围车辆使用短时恒加速度近似，极端交互行为可能超出假设；
- QP 硬不可行时需要松弛车辆约束，因此不能对所有初始状态宣称安全；
- 当前没有使用 CLF、MPC、HJ 可达性或学习式安全价值函数；
- 已有结果集中在一个 TD3 检查点和 100 个追尾场景，泛化与统计显著性仍需验证。

后续可将有限候选与在线短时前视替换为连续安全价值函数或 Hamilton–Jacobi/CBVF 引导，在保留 HOCBF-QP 实时安全过滤接口的同时，形成连续的逃逸方向生成方法。

## 12. 与原始 Adv-BMT 的关系

Adv-BMT 原项目用于生成安全关键交通场景。本扩展使用其追尾场景评估已经训练完成的 TD3，并在执行阶段增加独立 RTA 插件。原项目介绍、数据生成、模型训练和引用信息请参阅根目录 `readme.md`。

