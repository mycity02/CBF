# FSM-supervised two-dimensional HOCBF runtime assurance

This runtime-assurance (RTA) layer protects a trained TD3 policy without changing the policy, training loop, reward, scenario generator, or MetaDrive source. The native evaluator remains `eval_td3_policy.py`; the isolated RTA entrypoint is `eval_td3_policy_rta.py`.

The current default plugin is `fsm_hocbf_qp`. The v1 enumerative implementation remains available as `predictive_escape_cbf2d` for rollback and ablation.

## Runtime placement

```text
TD3 predict()
  -> nominal [steering, throttle/brake]
  -> threat activation and directional margins
  -> persistent FSM state
  -> one state-dependent reference control
  -> one two-dimensional HOCBF-QP
  -> safe [steering, throttle/brake]
  -> MetaDrive env.step()
```

The plugin reads the current ego state, surrounding traffic, and local road clearances. It does not change TD3 observations or rewards.

## Method

The FSM states are `nominal`, `brake`, `escape_left`, `escape_right`, and `recovery`. Entry/exit hysteresis, dwell time, consecutive-clear steps, and a recovery interval suppress left/right chattering. Safety escalations are not delayed by the dwell timer. The brake reference scales linearly with current front-barrier risk and, after 10 persistent brake steps, ramps to the configured maximum over 20 steps. The HOCBF-QP remains the hard safety layer. Front-triggered lateral escape is disabled by default until route-feasibility information is available; rear-triggered escape remains active.

Each state supplies one continuous acceleration/curvature reference. The same active vehicle and road HOCBF constraints are then used in a single two-variable QP:

\[
u^*=\arg\min_u \|u-u_q^{\mathrm{ref}}\|_W^2.
\]

The left and right states are maneuver modes, not fixed actions. Directional margins combine current road width with nearby traffic at a virtual lateral target. Threat activation uses only current HOCBF values, and escape curvature is an algebraic feedback on current heading error; the FSM path performs no finite-horizon motion prediction or candidate rollout.

The legacy plugin solves three QPs per active step for nominal, left, and right references and selects among short-horizon candidate rollouts. The FSM plugin instead preserves one mode across steps and solves one QP for the active mode. It therefore removes per-step parallel candidate evaluation, not discrete supervisory structure itself.

Road barriers are evaluated every step for telemetry and enter the QP when the current road barrier reaches its event threshold. If hard constraints have no common solution, the prioritized solver keeps active road constraints mandatory whenever physically feasible and relaxes normalized vehicle constraints. Telemetry reports hard feasibility, fallback mode, and normalized violation.

Formal claims must be limited to the local kinematic model, bounded traffic-acceleration assumptions, sampled updates, and states for which the HOCBF inequalities are feasible.

## Run

```powershell
python eval_td3_policy_rta.py `
  --ckpt_path <checkpoint.zip> `
  --eval_data_dir <rear_attack_scenario_directory> `
  --num_episodes 100 `
  --horizon 100 `
  --rta_plugin fsm_hocbf_qp `
  --rta_config cfgs/rta_fsm_hocbf_default.json `
  --out analysis_outputs/rta_fsm_hocbf_eval.json
```

Use the legacy enumerative ablation with:

```powershell
python eval_td3_policy_rta.py `
  --ckpt_path <checkpoint.zip> `
  --eval_data_dir <scenario_directory> `
  --rta_plugin predictive_escape_cbf2d `
  --rta_config cfgs/rta_cbf2d_side_vru_completion.json
```

Disable RTA in the same evaluator with `--rta_plugin none`.

## Telemetry

Per-step output includes:

- current/previous FSM state, transition flag and reason, and state age;
- left/right directional margins and the state-dependent reference control;
- QP solve count (one on active FSM steps, zero on pass-through steps);
- active obstacle count, HOCBF values/residuals, feasibility, and fallback mode;
- nominal and safe physical controls and normalized MetaDrive actions.

Per-episode output includes intervention count/rate, infeasible steps, QP solves, FSM transitions, barrier minima, mode counts, collision direction, reward, completion, speed, and latency.

## Current paired result

On the same native-scene-trained TD3 checkpoint and 100 rear-attack scenarios:

| Method | Reward | Completion | Collision | Out of road | Arrival | Mean speed |
|---|---:|---:|---:|---:|---:|---:|
| No RTA | 28.786 | 0.638 | 17% | 29% | 64% | 11.601 m/s |
| v17 enumerative HOCBF-QP | 33.321 | 0.693 | 5% | 28% | 73% | 10.564 m/s |
| Previous FSM | 32.392 | 0.678 | 6% | 26% | 74% | 9.881 m/s |
| Fixed-brake prediction-free FSM-HOCBF-QP | **34.562** | 0.623 | 7% | **18%** | **75%** | 9.626 m/s |
| Adaptive-brake prediction-free FSM-HOCBF-QP | 34.440 | 0.617 | **5%** | 21% | **75%** | **10.237 m/s** |

The raw event flags can overlap. Exclusive adaptive-brake outcomes are 75% success, 18% road violation, 5% collision, and 2% timeout/other. Relative to the fixed-brake version, mean speed rises by 6.35% and collision termination falls by two percentage points, while reward, completion, and road-violation termination change by -0.35%, -1.02%, and +3 percentage points. This is a measured tradeoff, not metric-wise dominance. The filter averaged 5.61 ms with a 36.70 ms P95 and no misses of the 100 ms control deadline.

There are 26 degenerate one-step scenarios with reference routes shorter than 2 m. On the remaining 74 dynamic scenarios, the adaptive-brake version obtains reward 44.784, completion 0.771, collision 6.76%, out-of-road 24.32%, and arrival 66.22%. Against v17, it has higher reward, equal collision rate, lower out-of-road rate, and higher arrival, but 0.010 lower completion.

This is not evidence that every metric dominates v17: the collision tradeoff, multi-seed confidence intervals, and parameter sensitivity remain open evaluation items.

## Tests

```powershell
python -m pytest tests/test_runtime_assurance_cbf2d.py `
  tests/test_runtime_assurance_metrics.py -q
```

Tests cover the QP projection, road-priority fallback, activation gating, squeeze and side-obstacle behavior, FSM dwell/hysteresis/recovery, episode reset, the single-QP contract, the MetaDrive adapter, and runtime metrics.
