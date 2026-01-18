# MDP Compliance Cheatsheet

- **State spaces**: 4D/12D/14D variants (see `env/sumo_env.py` state builders) selected via config state augmentation.
- **Queue mode**: `distinct_cycle` enforced for MDP compliance (`env/sumo_env.py` queue counting).
- **Slip lanes**: Excluded from control/state per lane group definitions (`env/sumo_env.py`: lane grouping).
- **Rewards**: Base/enhanced/PCU-weighted with optional `reward_exponent` (`env/sumo_env.py` reward computation; keys `use_enhanced_reward`, `use_pcu_weighted_wait`, `reward_exponent`).
- **Actions**: Action table normalized to `rho_ns` (`scripts/config_normalization.normalize_action_table_schema`); constraints encoded in `env/sumo_env.py` action defs.
- **Parameter sharing**: Multi-agent parameter sharing with state augmentation driven by config (`rl/` controllers + `env/sumo_env.py`).
- **Route pool selection**: Deterministic by seed and episode index (`env/sumo_env.py::_select_route_from_pool`).
