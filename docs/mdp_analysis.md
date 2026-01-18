# MDP Definition Analysis: SUMOEnv Traffic Signal Control

This document defines the Markov Decision Process (MDP) formulation from the `SUMOEnv` class implementation.

**Updated:** 2026-01-16 (SMDP v5 - 14D State)

---

## State Space ($\mathcal{S}$)

The state space uses **14 dimensions** (`state_dim: 14`), built by [`_build_state_vector`](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1069-1090).

### State Vector Structure (14 Dimensions)

| Index | Feature | Description | Range | Source |
|-------|---------|-------------|-------|--------|
| 0 | $Q_{NS,L}$ | Queue count (NS Left-turn) | $[0, \infty)$ | Local per-TLS |
| 1 | $Q_{NS,T}$ | Queue count (NS Through) | $[0, \infty)$ | Local per-TLS |
| 2 | $Q_{EW,L}$ | Queue count (EW Left-turn) | $[0, \infty)$ | Local per-TLS |
| 3 | $Q_{EW,T}$ | Queue count (EW Through) | $[0, \infty)$ | Local per-TLS |
| 4 | $W_{NS,L}$ | Waiting time (NS Left-turn) | $[0, \infty)$ sec | Local per-TLS |
| 5 | $W_{NS,T}$ | Waiting time (NS Through) | $[0, \infty)$ sec | Local per-TLS |
| 6 | $W_{EW,L}$ | Waiting time (EW Left-turn) | $[0, \infty)$ sec | Local per-TLS |
| 7 | $W_{EW,T}$ | Waiting time (EW Through) | $[0, \infty)$ sec | Local per-TLS |
| 8 | $\text{Occ}_N$ | Downstream occupancy (North) | $[0, 1]$ | Center TLS only* |
| 9 | $\text{Occ}_E$ | Downstream occupancy (East) | $[0, 1]$ | Center TLS only* |
| 10 | $\text{Occ}_S$ | Downstream occupancy (South) | $[0, 1]$ | Center TLS only* |
| 11 | $\text{Occ}_W$ | Downstream occupancy (West) | $[0, 1]$ | Center TLS only* |
| **12** | $\tilde{N}$ | **Normalized vehicle count** | $[0, 1]$ | **Global broadcast** |
| **13** | $\tilde{S}$ | **Normalized spillback scalar** | $[0, 1]$ | **Global broadcast** |

*\*Non-center TLS agents receive zeros for indices 8-11.*

### Global Broadcast Scalars (Index 12-13)

These two dimensions are **broadcast to ALL agents** to ensure the Markov property (SMDP v5):

```python
# From _step_multi() in sumo_env.py
n_present = traci.vehicle.getIDCount()
n_present_norm = min(1.0, float(n_present) / N_CAP)  # N_CAP = 10000

spill_scalar = ALPHA * sum(downstream_occupancy ** 2)  # ALPHA = 3.0
spill_scalar_norm = min(1.0, spill_scalar / (ALPHA * M))  # M = 4 (directions)
```

**Why needed:** The reward function depends on global variables ($N$, spillback). Without broadcasting, agents cannot observe what determines their reward → POMDP. Broadcasting ensures full observability → MDP.

### Code Implementation

```python
def _build_state_vector(self, tls_id: str, last_q_dir: np.ndarray, w_dir: np.ndarray,
                        n_present_norm: float = 0.0, spill_scalar_norm: float = 0.0) -> np.ndarray:
    occupancy = np.zeros(4, dtype=np.float32)
    if self._enable_downstream_occupancy and tls_id == self._center_tls_id:
        occupancy = self._read_downstream_occupancy()
    
    state = np.zeros(14, dtype=np.float32)  # 14D state
    state[0:4] = last_q_dir.astype(np.float32)   # Queue counts (local)
    state[4:8] = w_dir.astype(np.float32)        # Waiting times (local)
    state[8:12] = occupancy                       # Downstream occupancy (center only)
    state[12] = n_present_norm                    # Global vehicle count (broadcast)
    state[13] = spill_scalar_norm                 # Global spillback (broadcast)
    return state
```

---

## Action Space ($\mathcal{A}$)

### Discrete Action Space: 15 Actions

**Cycle Options**: `[60, 90, 120]` seconds  
**Split Options**: `[(0.30, 0.70), (0.40, 0.60), (0.50, 0.50), (0.60, 0.40), (0.70, 0.30)]`

$$|\mathcal{A}| = 3 \times 5 = 15$$

| Action ID | Cycle (s) | $\rho_{NS}$ | $\rho_{EW}$ | $g_{NS}$ (s) | $g_{EW}$ (s) |
|-----------|-----------|-------------|-------------|--------------|--------------|
| 0-4 | 60 | 0.30-0.70 | 0.70-0.30 | 18-42 | 42-18 |
| 5-9 | 90 | 0.30-0.70 | 0.70-0.30 | 27-63 | 63-27 |
| 10-14 | 120 | 0.30-0.70 | 0.70-0.30 | 36-84 | 84-36 |

---

## Reward Function ($R$) - SMDP v5

The reward is computed using **SMDP time-exposure formula** from [`compute_normalized_reward_smdp`](file:///c:/Users/Dell/GroupProject2/env/mdp_metrics.py):

### Formula

$$R = -\frac{W_{\text{global}}}{N \cdot t_{\text{ref}}} - \frac{\alpha \sum_{d} \text{Occ}_d^2}{M} \cdot \frac{\Delta t}{t_{\text{ref}}}$$

Where:
- $W_{\text{global}}$ = Total waiting time for **entire network** (sum of all TLS)
- $N$ = `n_present` = Current vehicle count in network (`traci.vehicle.getIDCount()`)
- $t_{\text{ref}} = 60$ seconds (reference time for scaling)
- $\alpha = 3.0$ (spillback weight)
- $\text{Occ}_d$ = Downstream occupancy for direction $d$
- $M = 4$ (number of directions)
- $\Delta t$ = Decision duration (cycle + transitions)

### Key Design Choices

1. **Global Reward:** All 9 TLS agents receive the **same reward** (cooperative setting).
2. **Demand-Invariant:** Dividing by $N$ prevents reward scale explosion with traffic demand.
3. **Time-Exposure:** Dividing by $t_{\text{ref}}$ penalizes per unit time, preventing "cycle hack" where longer cycles artificially reduce penalty frequency.
4. **Squared Spillback:** $\text{Occ}^2$ penalizes high occupancy more aggressively than linear.

### Code Implementation

```python
def compute_normalized_reward_smdp(
    w_global: float,
    n_present: int,
    downstream_occ: np.ndarray,
    t_step_value: float,
    alpha: float = 3.0,
    t_ref: float = 60.0
) -> float:
    N = max(1, n_present)
    wait_term = -w_global / (N * t_ref)
    
    spill_scalar = alpha * float(np.sum(downstream_occ ** 2))
    M = max(1, len(downstream_occ))
    spill_term = -(spill_scalar / M) * (t_step_value / t_ref)
    
    return wait_term + spill_term
```

---

## Removed Components (v2.0)

The following penalty terms existed in earlier versions but are **disabled in mainline**:

| Component | Status | Reason for Removal |
|-----------|--------|-------------------|
| Anti-flicker penalty ($\kappa$) | **REMOVED** | Adds non-Markovian dependency; squared spillback provides sufficient stability |
| Teleport penalty ($\lambda$) | **REMOVED** | Teleports are simulation artifacts, not agent decisions |
| Deadlock penalty ($\omega$) | **REMOVED** | Deadlocks should be prevented by network design, not penalized post-hoc |
| Threshold-linear spillback | **REMOVED** | Replaced by squared spillback for smoother gradient |

These remain in code for ablation studies but are set to 0 in production configs.

---

## MDP Summary

| Component | Definition |
|-----------|------------|
| **State Space** | $\mathcal{S} \subseteq \mathbb{R}^{14}$: Queue (4D) + Wait (4D) + Occupancy (4D) + Global (2D) |
| **Action Space** | $\mathcal{A} = \{0, 1, ..., 14\}$: 15 discrete actions (3 cycles × 5 splits) |
| **Transition** | Deterministic given SUMO; stochastic from route pool |
| **Reward** | SMDP time-exposure: $-W/(N \cdot t_{ref}) - \alpha\sum\text{Occ}^2 / M \cdot \Delta t / t_{ref}$ |
| **Discount** | $\gamma = 0.99$ |
| **Horizon** | Finite: `max_sim_seconds` (typically 1800s) |

---

## Appendix: State Variants

For ablation studies, the system supports multiple state dimensions via `state_dim` config:

| Variant | Dimensions | Description | Use Case |
|---------|------------|-------------|----------|
| 4D | 4 | Queue counts only | Single-TLS legacy |
| 12D | 12 | Queue + Wait + Occupancy | Multi-TLS without global |
| **14D** | **14** | **12D + Global broadcast** | **Mainline (SMDP v5)** |

Config keys:
- `env.sumo.state_dim`: 4, 12, or 14
- `env.sumo.enable_downstream_occupancy`: true/false
