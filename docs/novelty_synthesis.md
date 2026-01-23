# Key Technical Contributions & Novelty Points

> Defense-ready synthesis of technical novelty, strictly grounded in code.

**Updated:** 2026-01-16 (SMDP v5 - Synchronized with mainline spec)

---

## 1) Semi-Markovian Dynamics: Principled SMDP Formulation

### Problem Solved
Standard MDP-based RL traffic controllers assume uniform timesteps. When actions represent different cycle lengths (60s, 90s, 120s), fixed per-step discounting creates **temporal inconsistency**—the agent values 1-hour horizons differently depending on cycle choice, not actual performance.

### Implementation

| Component | Function | Code Reference |
|-----------|----------|----------------|
| Variable-duration actions | Actions encode cycle length | [sumo_env.py L140-144](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L140): `SumoActionDefinition(cycle_sec, rho_ns, rho_ew)` |
| Time-normalized reward | SMDP exposure formula | [mdp_metrics.py](file:///c:/Users/Dell/GroupProject2/env/mdp_metrics.py): `compute_normalized_reward_smdp()` |
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

## 2) Spillback Prevention via Squared Occupancy Penalty

### Problem Solved
Local intersection controllers optimize their own queues, causing **"green wave collapse"**—clearing traffic into already-saturated downstream links, triggering gridlock.

### Implementation

**State Vector Indices 8-11: Downstream Occupancy**
```python
# sumo_env.py - Center TLS observes downstream congestion
occupancy = self._read_downstream_occupancy()  # [N, E, S, W], range [0, 1]
```

**Squared Spillback Penalty (SMDP v5):**
```python
# mdp_metrics.py - compute_normalized_reward_smdp()
spill_scalar = alpha * np.sum(downstream_occ ** 2)
spill_term = -(spill_scalar / M) * (t_step / t_ref)
```

$$P_{\text{spillback}} = \alpha \sum_{d \in \{N,E,S,W\}} \text{Occ}_d^2$$

| Design Choice | Rationale |
|---------------|-----------|
| **Squared** (not linear) | Convex penalty provides smooth gradient from 0% occupancy |
| **No threshold** | Avoids cliff-edge behavior; penalizes any congestion proportionally |
| **Time-normalized** | Dividing by $t_{\text{ref}}$ prevents cycle-length gaming |

### Why This is Novel
- **Network-aware state**: The agent observes not just *incoming* queues but *downstream capacity*—enables **metering**.
- **Smooth gradient**: Squared function provides gradient signal even at low occupancy, preventing creep toward gridlock.

---

## 3) Global State Broadcast for Markov Property (SMDP v5)

### Problem Solved
The reward function depends on global variables (total vehicles $N$, total spillback). If agents only observe local state, the system becomes a **Dec-POMDP**—agents cannot predict their own reward.

### Implementation

**14D State = 12D Local + 2D Global Broadcast**

| Index | Feature | Scope |
|-------|---------|-------|
| 0-7 | Queue counts, Waiting times | Local per-TLS |
| 8-11 | Downstream occupancy | Center TLS only |
| **12** | `n_present_norm` = $\min(1, N/N_{\text{CAP}})$ | **Global broadcast** |
| **13** | `spill_scalar_norm` = normalized spillback | **Global broadcast** |

```python
# sumo_env.py - _step_multi()
n_present = traci.vehicle.getIDCount()
n_present_norm = min(1.0, float(n_present) / N_CAP)  # N_CAP = 10000

spill_scalar = ALPHA * sum(downstream_occ ** 2)
spill_scalar_norm = min(1.0, spill_scalar / (ALPHA * M))

# Broadcast to ALL agents
state[12] = n_present_norm
state[13] = spill_scalar_norm
```

### Why This is Novel
- **Restores Markov property**: Agents observe what determines their reward → MDP, not POMDP.
- **Enables credit assignment**: Agents can correlate actions with global outcomes.

---

## 4) Sensor Robustness: Distinct-Cycle Queue Counting

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

## 5) System-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    SEMI-MDP TRAFFIC CONTROL SYSTEM                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  State Space (14D)                     Action Space (15 discrete)    │
│  ├─ Local queues (4D)                  ├─ Cycle: [60, 90, 120]s      │
│  ├─ Waiting times (4D)                 └─ Split: 5 NS/EW ratios      │
│  ├─ Downstream occupancy (4D)          ════════════════════════      │
│  └─ Global broadcast (2D) ← NEW        = 3 × 5 = 15 actions         │
│                                                                      │
│  Function Approximator                 SMDP Mechanisms               │
│  ├─ Dueling DQN                        ├─ Time-aware γ               │
│  │   └─ V(s) + A(s,a) - mean(A)        ├─ SMDP exposure reward       │
│  └─ Double DQN target selection        └─ Per-transition γ storage   │
│                                                                      │
│  Safety & Robustness                                                 │
│  ├─ Squared spillback penalty (α∑Occ²)                               │
│  └─ Distinct-cycle queue counting                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Core Novelty Statement (Defense-Ready)

> This system advances beyond heuristic RL-for-traffic by implementing a **mathematically grounded Semi-MDP formulation**. The three pillars—time-aware discounting, SMDP exposure reward, and per-transition gamma storage—ensure that value estimates are temporally consistent regardless of action duration. The **14D state with global broadcast** restores the Markov property for multi-agent coordination, while **squared spillback penalty** provides smooth gradient signal for congestion prevention. Coupled with robust distinct-cycle queue sensing, this controller achieves what actuated and naive RL cannot: **principled optimization over variable-duration actions with network-level awareness**.

---

## Historical Components (Ablation Only)

The following were explored but **removed from mainline** due to complexity/redundancy:

| Component | Reason Removed |
|-----------|----------------|
| Anti-flicker penalty ($\kappa$) | Non-Markovian dependency; squared spillback provides sufficient stability |
| Threshold-linear spillback | Hard cutoff provides no gradient below threshold |
| Teleport penalty | Simulation artifact, not agent decision |
| Deadlock penalty | Should be prevented by design, not penalized post-hoc |
