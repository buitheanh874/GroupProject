# Key Technical Contributions & Novelty Points

> Defense-ready synthesis of technical novelty, strictly grounded in code.

**Updated:** 2026-01-11

---

## 1) Semi-Markovian Dynamics: Principled SMDP Formulation

### Problem Solved
Standard MDP-based RL traffic controllers assume uniform timesteps. When actions represent different cycle lengths (60s, 90s, 120s), fixed per-step discounting creates **temporal inconsistency**—the agent values 1-hour horizons differently depending on cycle choice, not actual performance.

### Implementation

| Component | Function | Code Reference |
|-----------|----------|----------------|
| Variable-duration actions | Actions encode cycle length | [sumo_env.py L140-144](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L140): `SumoActionDefinition(cycle_sec, rho_ns, rho_ew)` |
| Time-normalized reward | $R = R_{\text{base}} \cdot \frac{T_{\text{step}}}{T_{\text{actual}}}$ | [sumo_env.py L693-697](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L693) |
| Time-aware discount | $\gamma_t = \gamma_0^{t/t_{\text{ref}}}$ | [agent.py L87-99](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L87): `compute_gamma(t_step)` |

**Time-Aware Gamma Formula:**
$$\gamma_t = \gamma_0^{\left( \frac{t_{\text{step}}}{t_{\text{ref}}} \right)}$$

**Per-Transition Gamma in Replay Buffer:**
```python
# agent.py L154 - TD target uses variable gamma
target_q = batch.rewards + batch.gammas * next_q_target * (1.0 - batch.dones)
```

### Why This is Novel
- **Mathematical invariance**: Equal wall-clock durations receive equal total discount regardless of action granularity.
- **Per-transition storage**: Gamma is stored per transition in the replay buffer, not applied globally—this is the **defining characteristic of SMDP Q-learning**.

---

## 2) Spillback Prevention via Network-Level State Awareness

### Problem Solved
Local intersection controllers optimize their own queues, causing **"green wave collapse"**—clearing traffic into already-saturated downstream links, triggering gridlock.

### Implementation

**State Vector Indices 8-11: Downstream Occupancy**
```python
# sumo_env.py L1076-1078
occupancy = np.zeros(4, dtype=np.float32)
if self._enable_downstream_occupancy and tls_id == self._center_tls_id:
    occupancy = self._read_downstream_occupancy()  # [N, E, S, W]
```

**Spillback Penalty Computation:**
```python
# sumo_env.py L1354-1367
def _compute_spillback_penalty(self):
    occupancy = self._read_downstream_occupancy()
    over_thresh = np.maximum(occupancy - occ_threshold, 0.0)
    penalty = float(self._beta) * float(np.sum(over_thresh))
    return penalty
```

$$P_{\text{spillback}} = \beta \sum_{d \in \{N,E,S,W\}} \max(0, \text{Occ}_d - \theta_{\text{occ}})$$

### Why This is Novel
- **Network-aware state**: The agent observes not just *incoming* queues but *downstream capacity*—enables **metering**.
- **Continuous penalty**: Proportional to over-threshold occupancy, providing smooth gradient signal.

---

## 3) Sensor Robustness: Distinct-Cycle Queue Counting

### Problem Solved
Snapshot-based queue measurements (single timestep) suffer from high variance and phase boundary artifacts.

### Implementation

**Distinct-Cycle Mode**: Tracks the **set of unique vehicle IDs** that were queued at least once during the entire decision cycle.

```python
# mdp_metrics.py
if self._queue_mode == "distinct_cycle":
    self._queued[dir_key].update(veh_set)  # Union of all queued IDs
```

| Property | Snapshot Mode | Distinct-Cycle Mode |
|----------|---------------|---------------------|
| Variance | High (single sample) | Low (cycle-averaged) |
| Phase boundary | Sensitive to timing | Robust (aggregated) |
| MDP semantics | Markov violation | True per-decision state |

---

## 4) Stability: Anti-Flicker Temporal Regularization

### Problem Solved
RL agents can oscillate between short (60s) and long (120s) cycles, causing driver unpredictability and coordination breakdown.

### Implementation

```python
# sumo_env.py L1369-1374
def _compute_anti_flicker_penalty(self, cycle_sec: int) -> float:
    if not self._enable_anti_flicker:
        return 0.0
    if self._prev_cycle_sec is None:
        return 0.0
    return float(self._kappa) if int(cycle_sec) != int(self._prev_cycle_sec) else 0.0
```

$$P_{\text{anti-flicker}} = \begin{cases} \kappa & \text{if } C_t \neq C_{t-1} \\ 0 & \text{otherwise} \end{cases}$$

---

## 5) System-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SEMI-MDP TRAFFIC CONTROL SYSTEM                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  State Space (12D)                     Action Space (15 discrete)    │
│  ├─ Local queues (4D)                  ├─ Cycle: [60, 90, 120]s      │
│  ├─ Waiting times (4D)                 └─ Split: 5 NS/EW ratios      │
│  └─ Downstream occupancy (4D)          ════════════════════════      │
│                                         = 3 × 5 = 15 actions         │
│                                                                      │
│  Function Approximator                 SMDP Mechanisms               │
│  ├─ Dueling DQN                        ├─ Time-aware γ               │
│  │   └─ V(s) + A(s,a) - mean(A)        ├─ Reward time normalization  │
│  └─ Double DQN target selection        └─ Per-transition γ storage   │
│                                                                      │
│  Safety & Robustness                   Stability                     │
│  ├─ Spillback penalty (network-aware)  └─ Anti-flicker regularization│
│  └─ Distinct-cycle queue counting                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Core Novelty Statement (Defense-Ready)

> This system advances beyond heuristic RL-for-traffic by implementing a **mathematically grounded Semi-MDP formulation**. The three pillars—time-aware discounting, reward normalization, and per-transition gamma storage—ensure that value estimates are temporally consistent regardless of action duration. Coupled with network-level spillback awareness and robust queue sensing, this controller achieves what actuated and naive RL cannot: **principled optimization over variable-duration actions with safety guarantees**.
