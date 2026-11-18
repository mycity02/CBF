# CBF / RTA

面向自动驾驶仿真的二维控制障碍函数（CBF）运行时保障模块。在策略给出名义动作后，用高阶控制障碍函数（HOCBF）和二次规划（QP）修正纵向加速度与路径曲率。包含道路约束、与周边车辆的间距约束，以及用于后向风险处理的有限状态监督器。

本仓库独立发布控制器、仿真适配器、配置和单元测试，不包含数据集、训练模型或第三方仿真器源码。控制核心只依赖 NumPy。

## 安装与测试

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## 调用

`FSMHOCBF` 接收以米、秒和弧度表示的车辆及道路状态。输入控制量是纵向加速度（m/s²）和路径曲率（1/m）。

```python
import numpy as np
from cbf_rta import Control2D, FSMHOCBF, RoadGeometry, VehicleState2D

ego = VehicleState2D(
    name="ego", position=np.array([0.0, 0.0]), velocity=np.array([8.0, 0.0]),
    acceleration=np.zeros(2), heading=0.0, length=4.5, width=1.8,
)
road = RoadGeometry(heading=0.0, left_clearance=3.5, right_clearance=3.5)
controller = FSMHOCBF()
result = controller.filter(Control2D(acceleration=0.0, curvature=0.0), ego, [], road)
print(result.control, result.hard_feasible)
```

若已有 MetaDrive 环境，可使用 `cbf_rta.adaptive_plugin.FSMHOCBFPlugin`：在环境重置后调用 `plugin.reset(env)`，每步调用 `safe_action, info = plugin.filter_action(env, nominal_action)`，再将 `safe_action` 交给环境执行。开关插件不需要修改原策略的网络参数或奖励函数。

- `cbf_rta/`：控制器、有限状态监督、QP 求解器和适配器。
- `cfgs/`：基础、FSM 和后向避让示例参数。
- `tests/`：控制器、适配器和指标的单元测试。

本项目用于仿真研究。约束不可行、参考几何不准确或其他车辆行为超出建模假设时，控制器不保证无碰撞，也不应作为实车安全承诺。
