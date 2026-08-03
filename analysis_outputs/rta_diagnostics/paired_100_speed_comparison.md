# RTA Speed and Runtime Comparison

All variants use the same TD3 checkpoint, the same 100 rear-attack scenarios,
deterministic policy inference, a 100-step horizon, and CPU execution.

The MetaDrive control period is:

- physics step: 0.02 s
- decision repeat: 5
- control period and RTA deadline: 0.1 s (100 ms)

Vehicle speed is the norm of the ego vehicle world-frame velocity after every
environment step. Low speed means below 1 m/s and stopped means below 0.1 m/s.
RTA latency measures only `filter_action()` and excludes model inference and
MetaDrive physics.

## Overall results

| Variant | Reward | Completion | Collision | Out of road | Arrive | Speed (m/s) | Speed (km/h) | Low speed | Stopped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No RTA | 29.759 | 0.677 | 14% | 37% | 72% | 11.386 | 40.99 | 2.92% | 0.54% |
| v14 | 28.716 | 0.634 | 2% | 41% | 65% | 11.034 | 39.72 | 8.19% | 2.60% |
| v15 completion profile | 32.228 | 0.702 | 6% | 30% | 76% | 10.329 | 37.18 | 7.05% | 1.09% |
| v17 reward profile | 33.167 | 0.693 | 5% | 33% | 75% | 10.446 | 37.60 | 7.26% | 0.90% |

The all-step mean speed is affected by termination bias: v15 and v17 keep more
difficult, slower episodes alive instead of terminating early after an
out-of-road event. It should not be interpreted alone.

## Paired common-arrival comparison against No RTA

Only scenarios reached successfully by both No RTA and the compared RTA
variant are included.

| Variant | Common arrivals | New arrivals | Lost arrivals | No-RTA time (s) | RTA time (s) | Time delta (s) | No-RTA speed (m/s) | RTA speed (m/s) | Speed delta (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v14 | 56 | 9 | 16 | 2.414 | 2.332 | -0.082 | 5.743 | 6.079 | +0.335 |
| v15 | 68 | 8 | 4 | 3.007 | 3.104 | +0.097 | 6.890 | 6.750 | -0.140 |
| v17 | 66 | 9 | 6 | 2.933 | 3.036 | +0.103 | 6.809 | 6.680 | -0.129 |

Thus, on like-for-like successful scenarios, v15 has an approximately 2.0%
speed penalty and 3.2% time penalty; v17 has an approximately 1.9% speed
penalty and 3.5% time penalty. Both improve the number of successful arrivals.

## RTA computation latency

| Variant | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Over 100 ms | Mean budget use |
|---|---:|---:|---:|---:|---:|---:|---:|
| v14 | 95.02 | 34.06 | 390.55 | 636.45 | 2219.73 | 29.06% | 95.02% |
| v15 completion profile | 16.04 | 9.13 | 44.88 | 122.61 | 185.57 | 1.87% | 16.04% |
| v17 reward profile | 19.83 | 10.28 | 67.60 | 137.01 | 174.51 | 3.38% | 19.83% |

v14 does not reliably meet the 100 ms runtime deadline. v15 is the strongest
real-time profile. v17 remains substantially faster than the deadline on
average, but has a higher tail latency and deadline-miss rate than v15.

## Result files

- `paired_100_speed_no_rta.json`
- `paired_100_speed_rta_v14.json`
- `paired_100_speed_rta_v15_completion.json`
- `paired_100_speed_rta_v17_reward.json`
