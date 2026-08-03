# Two-dimensional CBF runtime-assurance plugin

This runtime-assurance (RTA) layer protects a trained TD3 policy without
changing the policy, training loop, scenario generator, or MetaDrive source.
The native evaluator remains `eval_td3_policy.py`. The isolated RTA entrypoint
is `eval_td3_policy_rta.py`.

## Runtime placement

```text
TD3 predict()
    -> nominal [steering, throttle/brake]
    -> predictive_escape_cbf2d plugin
    -> safe [steering, throttle/brake]
    -> MetaDrive env.step()
```

The plugin reads the current ego state, traffic vehicles, and local road
clearances. It does not change the observation or reward seen by TD3.

## Safety formulation

- Vehicles inside the sensor cap are activated only when a constant-acceleration
  preview enters a buffered collision corridor. Lateral traffic whose predicted
  footprint remains separated does not create a CBF constraint.
- Active vehicles use a fourth-root-normalized superellipse. It preserves the
  collision boundary while preventing distant polynomial terms from dominating.
- Vehicle and road constraints are converted to relative-degree-two HOCBF
  inequalities in physical acceleration and path curvature.
- A dependency-free two-variable convex QP projects the nominal control onto
  the safe input set.
- Nominal, left-escape, and right-escape references are evaluated with a
  finite-horizon preview. Once the nominal future becomes unsafe, the safer
  feasible lateral mode can be selected.
- If the hard constraints have no common solution, the plugin returns a
  hierarchical fallback: road constraints remain mandatory whenever physically
  feasible, while normalized vehicle constraints are relaxed. Telemetry records
  the fallback mode and whether mandatory constraints remained feasible.

The first version uses a local kinematic approximation and a constant-control
preview. Formal claims must therefore be limited to the assumed model,
bounded traffic acceleration, sampled update time, and states for which the
HOCBF constraints remain feasible.

## Run

Use the same checkpoint and scenario directory as the native evaluator:

```powershell
python eval_td3_policy_rta.py `
  --ckpt_path <checkpoint.zip> `
  --eval_data_dir <rear_attack_scenario_directory> `
  --num_episodes 100 `
  --horizon 100 `
  --rta_plugin predictive_escape_cbf2d `
  --rta_config cfgs/rta_cbf2d_default.json `
  --out analysis_outputs/rta_cbf2d_eval.json
```

Run the isolated entrypoint without the plugin:

```powershell
python eval_td3_policy_rta.py `
  --ckpt_path <checkpoint.zip> `
  --eval_data_dir <scenario_directory> `
  --rta_plugin none
```

Run the untouched native evaluator:

```powershell
python eval_td3_policy.py `
  --ckpt_path <checkpoint.zip> `
  --eval_data_dir <scenario_directory>
```

## Telemetry

Per episode, the RTA evaluator records:

- intervention count and rate;
- hard-infeasible step count;
- minimum vehicle and road barrier values;
- nominal/left/right mode counts;
- approximate first-contact direction;
- the original reward, cost, completion, and collision metrics.

Per-step barrier diagnostics are available from
`AdaptiveMetaDriveCBFPlugin.filter_action()` and include barrier values,
HOCBF residuals, nominal/safe controls, candidate scores, and feasibility.
They also include raw/active obstacle counts, normalized violation, fallback
mode, and mandatory-constraint feasibility.

## Tests

```powershell
python -m pytest tests/test_runtime_assurance_cbf2d.py -q
```

The tests cover exact two-variable QP projection, minimal intervention,
front/rear squeeze lateral escape, distant lateral-vehicle rejection,
predictive activation, road-priority fallback, side-vehicle direction
selection, and the external MetaDrive adapter contract.
