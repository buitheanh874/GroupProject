# MDP Compliance Map

- **Reward time-normalization (T_step)**: `env/mdp_metrics.py:compute_normalized_reward`, used in `env/sumo_env.py`.
- **Distinct-cycle queues & waiting**: `env/mdp_metrics.py:CycleMetricsAggregator`, consumed in `env/sumo_env.py`.
- **Fairness metrics (max/p95)**: aggregator `fairness_value`, wired via env config `fairness_metric`.
- **Spillback & anti-flicker penalties**: helpers in `env/mdp_metrics.py`, applied in `env/sumo_env.py`.
- **PCU/enhanced rewards**: `CycleMetricsAggregator.waiting_total` with `use_weights`/`exponent` flags in `env/sumo_env.py`.
- **Action table validation**: `scripts/validation.py` (used by `scripts/common.py`).
- **Time-aware gamma (SMDP)**: `rl/agent.py` and `scripts/train.py`.
