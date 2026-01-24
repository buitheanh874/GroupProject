# Evidence Map - RL Traffic Signal Control Audit

## A) Evidence Table

| Section | Claim | Evidence (file:function/class:lines or snippet) | Notes/Assumptions |
|---------|-------|------------------------------------------------|-------------------|
| **Environment/Simulator** | SUMO microscopic traffic simulator with TraCI interface | `env/sumo_env.py:SUMOEnv:203-363` | Uses traci Python API |
| | 9-intersection grid network (BIGNET) | `configs/train_1.yaml:15-16` - `tls_ids: ["J0","J1","J2","J3","J4","J6","J7","J14","J17"]` | 3×3 grid with center at J0 |
| | Simulation step 1.0s | `configs/train_1.yaml:26` - `step_length_sec: 1.0` | Fixed step granularity |
| **MDP/MARL Formulation** | Semi-MDP (SMDP) with variable decision duration | `env/mdp_metrics.py:135-186` - `compute_normalized_reward_smdp()` comment: "Cycle is an action → decision duration (Δt) varies" | Each action = cycle execution |
| | Multi-agent (9 TLS agents, decentralized) | `env/sumo_env.py:216` - `self._tls_ids = [str(x) for x in config.tls_ids]` | Each TLS makes independent decisions |
| | Parameter sharing across agents | `rl/agent.py:47-57` - Single DuelingDQN network for all agents | Shared online_net and target_net |
| **State Space** | 14-dimensional state vector per agent | `env/sumo_env.py:1082-1116` - `_build_state_vector()` | Verified in `configs/train_1.yaml:57` |
| | [0:4] Queue counts (N,E,S,W) | `env/sumo_env.py:1111` - `state[0:4] = last_q_dir` | Distinct vehicles queued per cycle |
| | [4:8] Waiting times (N,E,S,W) | `env/sumo_env.py:1112` - `state[4:8] = w_dir` | Accumulated veh-sec |
| | [8:12] Downstream occupancy (N,E,S,W) | `env/sumo_env.py:1113` - `state[8:12] = occupancy` | [0,1] fractional |
| | [12] n_present_norm (global broadcast) | `env/sumo_env.py:1114` - `state[12] = float(n_present_norm)` | Vehicles currently in network |
| | [13] spill_scalar_norm (global broadcast) | `env/sumo_env.py:1115` - `state[13] = float(spill_scalar_norm)` | Global spillback penalty |
| | State normalization (z-score) | `env/normalization.py`, `configs/norm_turn801010.json` | 14D mean/std from 7560 samples |
| **Action Space** | 15 discrete actions (3 cycles × 5 splits) | `configs/train_1.yaml:35-41` - `cycle_options_sec: [60,90,120]`, `action_splits: [[0.30,0.70],...,[0.70,0.30]]` | 3×5=15 combinations |
| | Action defines (cycle_sec, ρ_NS, ρ_EW) | `env/sumo_env.py:140-144` - `SumoActionDefinition` dataclass | Split ratio ∈ {0.3,0.4,0.5,0.6,0.7} |
| | Min green constraint: g_min=10s | `configs/train_1.yaml:34` - `g_min_sec: 10` | Safety constraint |
| | Yellow phase: 3s, All-red: 2s | `configs/train_1.yaml:28-29` - `yellow_sec: 3`, `all_red_sec: 2` | Fixed transition phases |
| **Reward Function** | SMDP reward formula (v5) | `env/mdp_metrics.py:135-186` | Demand-invariant design |
| | R = -W/(N×t_ref) - (spill/M)×(Δt/t_ref) | `env/mdp_metrics.py:176,184` | W=wait_total, N=n_present, t_ref=60s |
| | Wait penalty: -W/(N×t_ref) | `env/mdp_metrics.py:176` - `wait_penalty = float(wait_total) / (float(n_veh) * t_ref_safe)` | Per-vehicle, time-scaled |
| | Spill penalty: -(α∑occ²/M)×(Δt/t_ref) | `env/mdp_metrics.py:184` - `spill_exposure = (float(spill_penalty)/float(M)) * (delta_t_safe/t_ref_safe)` | α=3.0 (train), α=1.0 (eval) |
| | t_ref = 60.0s | `configs/train_1.yaml:301` - `t_ref: 60.0` | Reference for time-aware γ |
| **Episode/Horizon** | Max simulation time: 1800s (train), 1500s (eval) | `configs/train_1.yaml:31`, `configs/eval.yaml:39` | Time-based termination |
| | Warmup: 300s subtracted from KPIs (eval) | `configs/eval.yaml:219` - `warmup: 300` | KPI correction excludes warmup |
| **Constraints** | Yellow phase: 3s (fixed) | `configs/train_1.yaml:28` | Before each direction switch |
| | All-red phase: 2s (fixed) | `configs/train_1.yaml:29` | After yellow, before next green |
| | Minimum green: 10s | `configs/train_1.yaml:34` - `g_min_sec: 10` | Safety minimum |
| | ρ_min: 0.1 | `configs/train_1.yaml:33` - `rho_min: 0.1` | Min split ratio |
| **Training Procedure** | Dueling DQN with Double DQN update | `rl/agent.py:161-165` - DDQN target calculation | `next_actions = argmax(online)`, then `target.gather()` |
| | Time-aware gamma: γ = γ₀^(Δt/t_ref) | `rl/agent.py:88-100` - `compute_gamma()` | γ₀=0.99, t_ref=60s |
| | Hidden dims: [256, 256] | `configs/train_1.yaml:298` | 2-layer MLP |
| | Learning rate: 1e-4 | `configs/train_1.yaml:302` | Adam optimizer |
| | Batch size: 256 | `configs/train_1.yaml:303` | Experience replay |
| | Replay buffer: 200,000 | `configs/train_1.yaml:304` | Prioritized/uniform |
| | Target update freq: 5000 steps | `configs/train_1.yaml:305` | Hard update |
| | Epsilon: 0.60→0.05 | `configs/train_1.yaml:323-326` | Linear decay after warmup |
| | Curriculum: 9 phases (easy→medium→hard) | `configs/train_1.yaml:341-417` | 500/750/1000 veh/hr |
| | Parallel training: 10 workers | `configs/train_1.yaml:424-432` | Async experience collection |
| | Huber loss | `rl/agent.py:64` - `SmoothL1Loss if use_huber else MSELoss` | Robust to outliers |
| | Gradient clipping: 10.0 | `configs/train_1.yaml:306` | Prevent exploding gradients |
| **Evaluation Protocol** | Seen demands: 500, 750, 1000 veh/hr | `configs/eval.yaml:209-212` | Training demands |
| | Unseen demands: 1250 (from conversation history) | Inferred from user's previous requests | Generalization test |
| | 10 seeds per demand | `configs/eval.yaml:215` - `seeds: 10` | Statistical significance |
| | Horizon: 1500s, Warmup: 300s | `configs/eval.yaml:218-219` | Corrected metrics |
| **Baselines** | Fixed-time controller | `controllers/fixed_time.py:FixedTimeController` | Uses `fixed_action_id` |
| | Max Pressure controller | `controllers/max_pressure.py:MaxPressureSplitController` | Queue-based selection |
| | Actuated controller (gap-out) | `controllers/actuated.py:ActuatedController` | Vehicle-actuated |
| | Webster controller | `controllers/webster.py:WebsterController` | Webster formula timing |
| **Metrics** | Average wait time (corrected) | `env/kpi.py:26,277` - `avg_wait_time_corr` | Teleported vehicles capped |
| | Average travel time (corrected) | `env/kpi.py:27,278` - `avg_travel_time_corr` | Teleported vehicles capped |
| | Throughput (corrected) | `env/kpi.py:25,292` - `throughput_corr` | Arrived (non-teleported) / steps |
| | Completion rate | `env/kpi.py:23,290` - `completion_rate = arrived/departed` | Fraction completed |
| | Teleport rate | `env/kpi.py:20,238` - `teleport_rate = unique_teleport/departed` | Failure indicator |
| | Queue length (average) | `env/kpi.py:15,233` - `avg_queue` | Mean across timesteps |
| | P95 wait time | `env/kpi.py:17,279` - `p95_wait_time_corr` | 95th percentile |
| **Seeding/Reproducibility** | Global seed: 42 | `configs/train_1.yaml:3` - `seed: 42` | Random state control |
| | Seed propagation to env, agent, numpy | `rl/utils.py:set_global_seed` | Deterministic training |
| | Worker-specific port offsets | `configs/train_1.yaml:427` - `base_port: 9500` | Parallel isolation |
| **Logging/Plots** | Training logs: `logs/1/` | `configs/train_1.yaml:420` | Episode rewards, metrics |
| | Model checkpoints: `models/1/` | `configs/train_1.yaml:421` | Best and latest models |
| | Eval results CSV | `eval_turn801010_baselines_s10.csv` root | 122 rows: policy,demand,seed,metrics |

## B) Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dueling DQN Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│  Input: State (14D)                                             │
│    ↓                                                            │
│  Feature Network: Linear(14→256) → ReLU → Linear(256→256) → ReLU│
│    ↓                                                            │
│  ┌──────────────┐    ┌───────────────────┐                      │
│  │ Value Head   │    │ Advantage Head    │                      │
│  │ Linear(256→1)│    │ Linear(256→15)    │                      │
│  └──────┬───────┘    └────────┬──────────┘                      │
│         │                     │                                 │
│         └──────────┬──────────┘                                 │
│                    ↓                                            │
│        Q(s,a) = V(s) + A(s,a) - mean(A)                        │
│                    ↓                                            │
│  Output: Q-values (15D)                                         │
└─────────────────────────────────────────────────────────────────┘
```
[evidence: rl/dueling_dqn.py:9-35]
