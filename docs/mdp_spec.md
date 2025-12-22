# MDP Specification (Report-aligned)

## Scope

Single intersection, 2-phase (NS/EW). Right-turn slip lanes are free-flow and excluded from MDP (not in state, not in reward).

## Time Scale

One decision step equals one signal cycle.

At cycle t:

1. The agent observes state s_t
2. The agent chooses action a_t (phase split)
3. The environment simulates one cycle under that split
4. The environment returns reward r_t and next state s_{t+1}

## Lane Grouping

Controlled lanes:

- LANES_NS_CTRL: incoming lanes (straight + left) on North–South axis controlled by TLS
- LANES_EW_CTRL: incoming lanes (straight + left) on East–West axis controlled by TLS

Slip lanes (right-turn free-flow):

- LANES_RT_NS: right-turn slip lanes for N/S approaches
- LANES_RT_EW: right-turn slip lanes for E/W approaches

Slip lanes are excluded from state and reward.

## State

Fixed-length vector:

state_raw = [q_NS, q_EW, w_NS, w_EW] with shape (4,)

Queue length:

- q_NS: number of halting vehicles on LANES_NS_CTRL (v < 0.1 m/s)
- q_EW: number of halting vehicles on LANES_EW_CTRL (v < 0.1 m/s)

Waiting time per cycle (vehicle-seconds):

- During each simulation second within the cycle, compute q_NS_step and q_EW_step using halting numbers
- w_NS = sum over the cycle of q_NS_step
- w_EW = sum over the cycle of q_EW_step

## State Normalization

Z-score for each component x:

x_norm = (x - mu_x) / (sigma_x + eps)

Then clip all components:

s_norm = clip(s_norm, -5, 5)

mu_x and sigma_x must be estimated from baseline fixed-time runs and stored in config.

## Action

Cycle-based set phase split with fixed cycle length C = 60s (default).

rho_NS + rho_EW = 1 and rho_NS, rho_EW >= rho_min

Default rho_min = 0.1.

Discrete action set (action_id 0..4):

- a0: (0.30, 0.70)
- a1: (0.40, 0.60)
- a2: (0.50, 0.50)
- a3: (0.60, 0.40)
- a4: (0.70, 0.30)

## Reward (Report-aligned)

Total waiting time within the cycle:

W_t = w_NS(t) + w_EW(t)

Reward per cycle:

r_t = - W_t

No reward difference term (do not use W_t - W_{t-1}). No fairness term in reward.

## Fairness

Fairness is future work and not part of the reward.

Anti-starvation is enforced by rho_min, ensuring each phase has a minimum green time each cycle.

## RL Default Hyperparameters

- gamma = 0.98
- learning_rate = 1e-3
- batch_size = 64
- replay_buffer_size = 100000
- target_update_freq = 1000
- eps_start = 1.0
- eps_end = 0.05
- eps_decay_steps = 50000
# MDP Specification — Single-Intersection Traffic Signal Control (SUMO)

**Purpose.** This document defines the Markov Decision Process (MDP) used in our project for adaptive traffic signal control at a **single 4-leg intersection** in **SUMO/TraCI**, aligned with the team report and implementation plan.

**Key simplification (implementation-friendly).** **Right-turn slip lanes are modeled as free-flow** (not controlled by the traffic light), and **are excluded from both state and reward**. They may be included in KPI reporting only.

---

## 0) Scope and assumptions

- **Controlled intersection type:** 4-way intersection, aggregated into **two main directions**:
  - **North–South (NS)**
  - **East–West (EW)**
- **Signal logic:** **two-phase** (NS green vs EW green). The agent does not toggle phases every step; instead it selects a **cycle-level phase split**.
- **Right-turn slip lanes:** always permissive (free-flow); excluded from MDP.
- **Yellow / all-red:** if modeled, they are **fixed constants** and **not part of the action**. For the first demo, yellow can be set to 0s (must be stated clearly).

---

## 1) Lane grouping (must be configured from `.net.xml`)

We separate lanes into **controlled lanes** and **slip lanes**.

### 1.1 Controlled lanes (controlled by TLS)

- `LANES_NS_CTRL`: all incoming lanes on the NS axis that are subject to the TLS (through + left-turn if present).
- `LANES_EW_CTRL`: all incoming lanes on the EW axis that are subject to the TLS (through + left-turn if present).

**Placeholder example (replace with actual lane IDs):**
```python
LANES_NS_CTRL = ["N_in_0", "N_in_1", "S_in_0", "S_in_1"]
LANES_EW_CTRL = ["E_in_0", "E_in_1", "W_in_0", "W_in_1"]
```

### 1.2 Slip lanes (right-turn free-flow)

- `LANES_RT_NS`: right-turn slip lanes on N/S approaches
- `LANES_RT_EW`: right-turn slip lanes on E/W approaches

**Rule:** slip lanes may be used for KPI (throughput, travel time, etc.) but **must not contribute to state/reward**.

---

## 2) Time scale (cycle-based MDP)

- **One decision step = one signal cycle** (cycle-based control).
- At decision step (cycle) `t`:
  1. Observe state `s_t` (recommended: snapshot at end of previous cycle / end of current cycle before action is applied, but must be consistent).
  2. Choose action `a_t` (phase split).
  3. Run the simulation for the full cycle using that split.
  4. Compute reward `r_t` at the end of the cycle and produce `s_{t+1}`.

---

## 3) State definition

### 3.1 State vector (raw)

We use a 4D state capturing congestion by axis:

\[
s_t = [q_{NS}, q_{EW}, w_{NS}, w_{EW}]
\]

- **Queue length (halting vehicles)**
  - `q_NS`: number of halting vehicles on `LANES_NS_CTRL`
  - `q_EW`: number of halting vehicles on `LANES_EW_CTRL`
- **Waiting time (vehicle-seconds per cycle)**
  - `w_NS`: cumulative waiting time on controlled NS lanes during the cycle
  - `w_EW`: cumulative waiting time on controlled EW lanes during the cycle

### 3.2 How to compute `q_*` and `w_*` (TraCI-friendly)

A vehicle is considered **queued** if its speed is below:
- `v_th = 0.1 m/s` (SUMO “halting” threshold is compatible via `getLastStepHaltingNumber`).

Per simulation step (recommended 1s):
```python
q_NS_step = sum(traci.lane.getLastStepHaltingNumber(l) for l in LANES_NS_CTRL)
q_EW_step = sum(traci.lane.getLastStepHaltingNumber(l) for l in LANES_EW_CTRL)
```

**Snapshot (queue):**
- At the **end of the cycle**: `q_NS = q_NS_step`, `q_EW = q_EW_step`.

**Cycle waiting time (vehicle-seconds):**
- Accumulate across the cycle of length `C` seconds:
```python
w_NS += q_NS_step
w_EW += q_EW_step
```

This implements:
\[
w_{NS} = \sum_{	au=1}^{C} q_{NS}^{step}(	au), \quad
w_{EW} = \sum_{	au=1}^{C} q_{EW}^{step}(	au)
\]

### 3.3 State ordering (strict)

The order **must be identical** across `SUMOEnv`, normalization, RL agent, and any UI:
```text
[q_NS, q_EW, w_NS, w_EW]
```

---

## 4) State normalization (recommended)

Because the scales differ, apply **z-score + clipping** for stable learning.

For each component \( x \in \{q_{NS}, q_{EW}, w_{NS}, w_{EW}\} \):
\[
x_{norm} = rac{x - \mu_x}{\sigma_x + \epsilon}
\]

Then clip:
\[
s_{norm} = clip(s_{norm}, -5, 5)
\]

- `mu_x`, `sigma_x` should be estimated from **baseline fixed-time runs** (multiple scenarios), logged per cycle, then saved in a config file.
- Recommended `epsilon = 1e-8`.

---

## 5) Action space (discrete phase split)

### 5.1 Meaning

Action chooses the **green split** between NS and EW within one fixed cycle.

- Cycle length: `C = 60s`
- Split constraints:
  - `rho_NS + rho_EW = 1`
  - `rho_NS, rho_EW >= rho_min`
  - Default `rho_min = 0.1` → `g_min = 6s`

Green times:
\[
g_{NS} = 
ho_{NS} \cdot C,\quad g_{EW} = 
ho_{EW} \cdot C
\]

### 5.2 Discrete action set

We use 5 discrete actions (mapped by integer `action_id`):

| action_id | (rho_NS, rho_EW) |
|---:|:---:|
| 0 | (0.30, 0.70) |
| 1 | (0.40, 0.60) |
| 2 | (0.50, 0.50) |
| 3 | (0.60, 0.40) |
| 4 | (0.70, 0.30) |

---

## 6) Transition dynamics (environment step)

`SUMOEnv.step(action_id)` performs:

1. Map `action_id → (rho_NS, rho_EW)`
2. Compute `g_NS`, `g_EW`
3. Execute TLS schedule for one cycle:
   - Set NS green → simulate `g_NS` seconds (1s stepping recommended)
   - Optional fixed yellow/all-red (constant) → simulate `t_yellow` seconds
   - Set EW green → simulate `g_EW` seconds
   - Optional fixed yellow/all-red (constant) → simulate `t_yellow` seconds
4. During each 1s sim step:
   - compute `q_NS_step`, `q_EW_step`
   - accumulate `w_NS`, `w_EW`
5. End-of-cycle:
   - snapshot `q_NS`, `q_EW`
   - compute reward `r_t`
   - return `(s_{t+1}, r_t, done, info)`

**Important:** Slip lanes remain permissive and unchanged by action.

---

## 7) Reward function (report-aligned)

We optimize **negative total waiting time per cycle** (not a delta).

Total cycle waiting time:
\[
W_t = w_{NS}(t) + w_{EW}(t)
\]

Reward:
\[
r_t = -W_t
\]

**Exclusions:** Only controlled lanes (`LANES_NS_CTRL`, `LANES_EW_CTRL`) contribute to `W_t`. Slip lanes are excluded.

---

## 8) Fairness / anti-starvation

- **No explicit fairness term** is included in the default reward (treated as future work).
- Anti-starvation is enforced via the minimum green constraint `rho_min` (each direction receives at least `g_min` every cycle).

---

## 9) Episode definition

- One episode consists of multiple cycles (e.g., 1 hour simulation).
- Training and evaluation should include multiple demand scenarios (e.g., low/medium/high), each corresponding to one or more SUMO route files.

---

## 10) Default parameters (must stay consistent across code, configs, and report)

| Parameter | Default |
|---|---:|
| Cycle length `C` | 60 s |
| Queued speed threshold `v_th` | 0.1 m/s |
| Minimum split `rho_min` | 0.1 |
| Yellow time `t_yellow` | fixed constant (e.g., 2 s) or 0 s for initial demo |
| Discount factor `gamma` | 0.98 |

### Suggested RL defaults (for config sync)

| Hyperparameter | Default |
|---|---:|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Replay buffer size | 100000 |
| Target update frequency | 1000 steps |
| Epsilon-greedy | start=1.0, end=0.05, decay_steps=50000 |

---

## 11) Implementation checklist (quick)

- [ ] Lane IDs are correctly populated for `LANES_NS_CTRL`, `LANES_EW_CTRL`, and slip lanes.
- [ ] Slip lanes are excluded from `state` and `reward`.
- [ ] `state` ordering is exactly `[q_NS, q_EW, w_NS, w_EW]`.
- [ ] Reward uses **absolute** `-W_t` (not difference).
- [ ] `rho_min` is enforced in action design/config.
- [ ] Normalization parameters `(mu, sigma)` are computed from baseline logs and stored in config.

