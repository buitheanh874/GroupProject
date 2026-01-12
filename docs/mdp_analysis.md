# MDP Definition Analysis: SUMOEnv Traffic Signal Control

This document reverse-engineers the Markov Decision Process (MDP) formulation from the `SUMOEnv` class implementation for the thesis Methodology section.

**Updated:** 2026-01-11

---

## State Space ($\mathcal{S}$)

The state space uses **12 dimensions** (`state_dim: 12`), built by the [`_build_state_vector`](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1072-1084) method:

```python
def _build_state_vector(self, tls_id: str, last_q_dir: np.ndarray, w_dir: np.ndarray) -> np.ndarray:
    occupancy = np.zeros(4, dtype=np.float32)
    if self._enable_downstream_occupancy and tls_id == self._center_tls_id and len(self._downstream_links) > 0:
        occupancy = self._read_downstream_occupancy()
    
    state = np.zeros(12, dtype=np.float32)
    state[0:4] = last_q_dir.astype(np.float32)   # Queue counts
    state[4:8] = w_dir.astype(np.float32)        # Waiting times
    state[8:12] = occupancy                       # Downstream occupancy
    return state
```

### State Vector Structure (12 Dimensions)

| Index | Feature | Description | Range |
|-------|---------|-------------|-------|
| 0 | $Q_N$ | Queue count (North approach) | $[0, \infty)$ |
| 1 | $Q_E$ | Queue count (East approach) | $[0, \infty)$ |
| 2 | $Q_S$ | Queue count (South approach) | $[0, \infty)$ |
| 3 | $Q_W$ | Queue count (West approach) | $[0, \infty)$ |
| 4 | $W_N$ | Cumulative waiting time (North) | $[0, \infty)$ seconds |
| 5 | $W_E$ | Cumulative waiting time (East) | $[0, \infty)$ seconds |
| 6 | $W_S$ | Cumulative waiting time (South) | $[0, \infty)$ seconds |
| 7 | $W_W$ | Cumulative waiting time (West) | $[0, \infty)$ seconds |
| 8 | $\text{Occ}_N$ | Downstream occupancy (North exit) | $[0, 1]$ |
| 9 | $\text{Occ}_E$ | Downstream occupancy (East exit) | $[0, 1]$ |
| 10 | $\text{Occ}_S$ | Downstream occupancy (South exit) | $[0, 1]$ |
| 11 | $\text{Occ}_W$ | Downstream occupancy (West exit) | $[0, 1]$ |

### Downstream Occupancy Only for Center TLS

From [`_build_state_vector`](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1077):
```python
if self._enable_downstream_occupancy and tls_id == self._center_tls_id and len(self._downstream_links) > 0:
    occupancy = self._read_downstream_occupancy()
```

**Rationale:**
1. **Configuration Scope**: The `downstream_links` config maps directions to edge/lane IDs only for the center TLS (J0 in the config).
2. **Spillback Detection**: Downstream occupancy is primarily used for **spillback penalty** computation.
3. **MARL Homogeneity**: For non-center agents, the 4 occupancy dimensions are zeros, maintaining consistent state shape.

### Queue Count Mode: `distinct_cycle`

The [`CycleMetricsAggregator`](file:///c:/Users/Dell/GroupProject2/env/mdp_metrics.py) uses `distinct_cycle` mode:

| Metric | `snapshot` (deprecated) | `distinct_cycle` |
|--------|------------------------|------------------|
| **Definition** | Count of vehicles halted at the *last simulation step* | Count of *unique vehicles* halted **at least once** during the cycle |
| **Variance** | High—single-step snapshots are noisy | Low—aggregates over entire decision cycle |
| **MDP Compliance** | Poor—doesn't capture cumulative state | Good—reflects true queue demand per decision epoch |

---

## Action Space ($\mathcal{A}$)

### Action Definition Structure

From [`SumoActionDefinition`](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L140-144):
```python
@dataclass
class SumoActionDefinition:
    cycle_sec: int      # Total cycle length in seconds
    rho_ns: float       # Green split ratio for NS direction (0 < ρ < 1)
    rho_ew: float       # Green split ratio for EW direction (ρ_ew = 1 - ρ_ns)
```

### Discrete Action Space: 15 Actions

**Cycle Options**: `[60, 90, 120]` seconds
**Split Options**: `[(0.30, 0.70), (0.40, 0.60), (0.50, 0.50), (0.60, 0.40), (0.70, 0.30)]`

$$|\mathcal{A}| = 3 \times 5 = 15$$

| Action ID | Cycle (sec) | $\rho_{NS}$ | $\rho_{EW}$ | $g_{NS}$ (sec) | $g_{EW}$ (sec) |
|-----------|-------------|-------------|-------------|----------------|----------------|
| 0 | 60 | 0.30 | 0.70 | 18 | 42 |
| 1 | 60 | 0.40 | 0.60 | 24 | 36 |
| 2 | 60 | 0.50 | 0.50 | 30 | 30 |
| 3 | 60 | 0.60 | 0.40 | 36 | 24 |
| 4 | 60 | 0.70 | 0.30 | 42 | 18 |
| 5-9 | 90 | ... | ... | ... | ... |
| 10-14 | 120 | ... | ... | ... | ... |

---

## Reward Function ($R$)

The reward is computed per decision cycle using [`compute_normalized_reward`](file:///c:/Users/Dell/GroupProject2/env/mdp_metrics.py):

### Base Formula

$$R = -\frac{W_{\text{total}} + P_{\text{fairness}} + P_{\text{spillback}} + P_{\text{anti-flicker}}}{T_{\text{step}}}$$

Where:
- $W_{\text{total}}$ = Total waiting time accumulated during the cycle
- $T_{\text{step}} = C + 2 \cdot t_{\text{yellow}} + 2 \cdot t_{\text{all-red}}$ (full phase duration)

### Time Normalization (`reward_time_normalize: true`)

From [sumo_env.py L693-697](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L693):
```python
if self._reward_time_normalize:
    if transition_total_sec > 0:
        reward = reward * t_step_value / decision_duration_sec
    else:
        reward = reward / decision_duration_sec
```

$$R_{\text{normalized}} = R \cdot \frac{T_{\text{step}}}{T_{\text{actual}}}$$

---

## Penalty Terms

### 1. Spillback Penalty ($P_{\text{spillback}}$)

From [`_compute_spillback_penalty`](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1354-1367):
```python
def _compute_spillback_penalty(self) -> float:
    occupancy = self._read_downstream_occupancy()
    occ_threshold = float(np.clip(self._occ_threshold, 0.0, 1.0))
    over_thresh = np.maximum(occupancy - occ_threshold, 0.0)
    penalty = float(self._beta) * float(np.sum(over_thresh))
    return penalty
```

$$P_{\text{spillback}} = \beta \sum_{d \in \{N,E,S,W\}} \max(0, \text{Occ}_d - \theta_{\text{occ}})$$

### 2. Anti-Flicker Penalty ($P_{\text{anti-flicker}}$)

From [`_compute_anti_flicker_penalty`](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1369-1374):
```python
def _compute_anti_flicker_penalty(self, cycle_sec: int) -> float:
    if not self._enable_anti_flicker:
        return 0.0
    if self._prev_cycle_sec is None:
        return 0.0
    return float(self._kappa) if int(cycle_sec) != int(self._prev_cycle_sec) else 0.0
```

$$P_{\text{anti-flicker}} = \begin{cases} \kappa & \text{if } C_t \neq C_{t-1} \\ 0 & \text{otherwise} \end{cases}$$

### 3. Teleport Penalty

From [sumo_env.py L686-688](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L686):
```python
if float(self._teleport_penalty_lambda) > 0.0:
    teleport_penalty = teleport_penalty_lambda * decision_teleport_count
    reward = reward - teleport_penalty
```

$$P_{\text{teleport}} = \lambda_{\text{tele}} \cdot n_{\text{teleported}}$$

---

## MDP Summary

| Component | Definition |
|-----------|------------|
| **State Space** | $\mathcal{S} \subseteq \mathbb{R}^{12}$: Queue counts (4D) + Waiting times (4D) + Downstream occupancy (4D) |
| **Action Space** | $\mathcal{A} = \{0, 1, ..., 14\}$: 15 discrete actions (3 cycles × 5 splits) |
| **Transition** | Deterministic given SUMO simulation; stochastic from route pool randomization |
| **Reward** | Negative normalized waiting time with penalty terms |
| **Discount** | $\gamma = 0.99$ (or time-aware: $\gamma_t = \gamma_0^{t/t_{\text{ref}}}$) |
| **Horizon** | Finite: 50 cycles or 3000 simulation seconds |
