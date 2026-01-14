# Traffic Signal Control RL Project: Formulas and Setup Guide

**Scope**: This document synthesizes all major formulas, hyperparameters, and setup instructions for the multi-agent Traffic Signal Control (TSC) Reinforcement Learning project using SUMO simulation. All values are extracted directly from the codebase with source references.

---

## Table of Contents

1. [Notation and Symbols](#notation-and-symbols)
2. [State / Observation Space](#state--observation-space)
3. [Action Space](#action-space)
4. [Reward Function](#reward-function)
5. [Training (DQN)](#training-dqn)
6. [Normalization](#normalization)
7. [Parallel Training](#parallel-training)
8. [Setup and Run](#setup-and-run)
9. [Troubleshooting](#troubleshooting)
10. [Appendix: Sources Inventory](#appendix-sources-inventory)
11. [Completeness Checklist](#completeness-checklist)

---

## Notation and Symbols

| Symbol | Meaning |
|--------|---------|
| `t_step` | Decision cycle duration including transitions (cycle + 2*yellow + 2*all_red) |
| `rho_ns`, `rho_ew` | Split ratios for NS and EW phases (sum = 1.0) |
| `g_ns`, `g_ew` | Green durations in seconds for NS and EW |
| `gamma` | Discount factor for RL |
| `epsilon` | Exploration rate for epsilon-greedy policy |
| `Q(s,a)` | Action-value function |

---

## State / Observation Space

**Dimension**: 12D (per TLS agent)

(source: [env/sumo_env.py:L1074-1086](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1074-1086))

```python
def _build_state_vector(self, tls_id: str, last_q_dir: np.ndarray, w_dir: np.ndarray) -> np.ndarray:
    occupancy = np.zeros(4, dtype=np.float32)
    if self._enable_downstream_occupancy and tls_id == self._center_tls_id and len(self._downstream_links) > 0:
        occupancy = self._read_downstream_occupancy()
    state = np.zeros(12, dtype=np.float32)
    state[0:4] = last_q_dir.astype(np.float32)   # Queue counts per direction
    state[4:8] = w_dir.astype(np.float32)         # Waiting sums per direction
    state[8:12] = occupancy                        # Downstream occupancy per direction
    return state
```

### Feature Table

| Index | Name | Meaning | Signal Source (SUMO/TraCI) | Normalization (default) |
|-------|------|---------|---------------------------|------------------------|
| 0 | `q_N` | Distinct vehicles queued (North approach) | `traci.lane.getLastStepVehicleIDs` + speed < halt_threshold | mean=15, std=12 |
| 1 | `q_E` | Distinct vehicles queued (East approach) | Same as above | mean=15, std=12 |
| 2 | `q_S` | Distinct vehicles queued (South approach) | Same as above | mean=15, std=12 |
| 3 | `q_W` | Distinct vehicles queued (West approach) | Same as above | mean=15, std=12 |
| 4 | `w_N` | Cumulative waiting time (North) | Accumulated per-vehicle-step during green phases | mean=150, std=120 |
| 5 | `w_E` | Cumulative waiting time (East) | Same as above | mean=150, std=120 |
| 6 | `w_S` | Cumulative waiting time (South) | Same as above | mean=150, std=120 |
| 7 | `w_W` | Cumulative waiting time (West) | Same as above | mean=150, std=120 |
| 8 | `occ_N` | Downstream edge occupancy (North) | `traci.edge.getLastStepOccupancy` | mean=0.25, std=0.20 |
| 9 | `occ_E` | Downstream edge occupancy (East) | Same as above | mean=0.25, std=0.20 |
| 10 | `occ_S` | Downstream edge occupancy (South) | Same as above | mean=0.25, std=0.20 |
| 11 | `occ_W` | Downstream edge occupancy (West) | Same as above | mean=0.25, std=0.20 |

(source: [configs/norm_curriculum_v3.json](file:///c:/Users/Dell/GroupProject2/configs/norm_curriculum_v3.json))

### Queue Counting Mode

- **Mode**: `distinct_cycle` (only supported mode)
- Tracks unique vehicle IDs that queued at least once during the decision cycle
- Speed threshold for "halted": `halt_speed_threshold = 0.1` m/s

(source: [env/sumo_env.py:L303-311](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L303-311))

---

## Action Space

**Total Actions**: 15 (3 cycles x 5 splits)

(source: [env/sumo_env.py:L1509-1563](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1509-1563))

### Cycle Options (seconds)

```yaml
cycle_options_sec: [60, 90, 120]
```

(source: [configs/train_1.yaml:L36](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L36))

### Split Ratios (rho_ns, rho_ew)

```yaml
action_splits:
  - [0.30, 0.70]
  - [0.40, 0.60]
  - [0.50, 0.50]
  - [0.60, 0.40]
  - [0.70, 0.30]
```

(source: [configs/train_1.yaml:L37-42](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L37-42))

### Action Table Mapping

| action_id | cycle_sec | rho_ns | rho_ew |
|-----------|-----------|--------|--------|
| 0 | 60 | 0.30 | 0.70 |
| 1 | 60 | 0.40 | 0.60 |
| 2 | 60 | 0.50 | 0.50 |
| 3 | 60 | 0.60 | 0.40 |
| 4 | 60 | 0.70 | 0.30 |
| 5 | 90 | 0.30 | 0.70 |
| 6 | 90 | 0.40 | 0.60 |
| 7 | 90 | 0.50 | 0.50 |
| 8 | 90 | 0.60 | 0.40 |
| 9 | 90 | 0.70 | 0.30 |
| 10 | 120 | 0.30 | 0.70 |
| 11 | 120 | 0.40 | 0.60 |
| 12 | 120 | 0.50 | 0.50 |
| 13 | 120 | 0.60 | 0.40 |
| 14 | 120 | 0.70 | 0.30 |

### Green Duration Calculation

```python
g_ns_raw = round(rho_ns * cycle_sec)
g_ns = max(min_green_sec, min(g_ns_raw, cycle_sec - min_green_sec))
g_ew = cycle_sec - g_ns
```

(source: [env/sumo_env.py:L1088-1099](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1088-1099))

### Constraints

| Parameter | Value | Source |
|-----------|-------|--------|
| `rho_min` | 0.1 | [configs/train_1.yaml:L33](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L33) |
| `g_min_sec` | 10 | [configs/train_1.yaml:L34](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L34) |
| `yellow_sec` | 3 | [configs/train_1.yaml:L28](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L28) |
| `all_red_sec` | 2 | [configs/train_1.yaml:L29](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L29) |

### Multi-Agent Mode

- **TLS IDs**: `["J0", "J1", "J2", "J3", "J4", "J6", "J7", "J14", "J17"]` (9 agents)
- All agents share the same action space (parameter sharing)
- All agents must select the same `cycle_sec` per decision step

(source: [configs/train_1.yaml:L15](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L15), [env/sumo_env.py:L794-797](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L794-797))

---

## Reward Function

### Core Formula (Simplified)

$$R = -\frac{W_{total}}{T} - \alpha \cdot \sum (Occ)^2 - P_{teleport} - P_{deadlock}$$

**Step 1**: Compute base reward with spillback penalty (source: [env/mdp_metrics.py:L135-164](file:///c:/Users/Dell/GroupProject2/env/mdp_metrics.py#L135-164))
```python
r_base = -wait_total / t_step - spill_penalty
```

**Step 2**: Apply teleport and deadlock penalties (source: [env/sumo_env.py:L660-680](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L660-680))
```python
reward = r_base - teleport_penalty - deadlock_penalty
```

> [!IMPORTANT]
> **The spillback penalty is NOT divided by T.** This ensures proper weighting: congestion penalty (~0.5-4.0 points) vs waiting time penalty (~0.3-1.0 points).

```python
def compute_normalized_reward(
    wait_total: float,
    t_step: float,
    decision_cycle_sec: float,
    spill_penalty: float = 0.0,
) -> float:
    denom = float(t_step) if float(t_step) > 0.0 else float(decision_cycle_sec)
    denom = max(1.0, float(denom))
    # R = -W/T - spill_penalty (spill NOT divided by T)
    return -float(wait_total) / float(denom) - float(spill_penalty)
```

### Spillback Penalty (Squared Occupancy)

Based on **Varaiya 2013 (Back-Pressure)** and **PressLight (KDD 2019)**:

```python
def _compute_spillback_penalty(self) -> float:
    occupancy = self._read_downstream_occupancy()  # 4D vector [N, E, S, W]
    penalty = self._alpha_spillback * np.sum(occupancy ** 2)
    return penalty
```

(source: [env/sumo_env.py:L1310-1330](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1310-1330))

**Why Squared Occupancy?**

| Approach | Formula | Issue |
|----------|---------|-------|
| **Threshold-based (OLD)** | `β * max(occ - 0.65, 0)` | Hard cutoff, no gradient below threshold |
| **Squared (NEW)** | `α * sum(occ²)` | Smooth gradient from 0%, convex penalty curve |

**Example penalties with α=1.0:**

| Downstream Occupancy | ∑(Occ)² | Spillback Penalty |
|---------------------|---------|-------------------|
| All at 30% | 4 × 0.09 = 0.36 | **0.36** |
| All at 50% | 4 × 0.25 = 1.0 | **1.0** |
| All at 70% | 4 × 0.49 = 1.96 | **1.96** |
| All at 100% (gridlock) | 4 × 1.0 = 4.0 | **4.0** |

### t_step Calculation

```python
t_step = cycle_sec + 2 * yellow_sec + 2 * all_red_sec
```

(source: [env/sumo_env.py:L658](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L658))

### wait_total Calculation

Accumulated waiting time = sum over all vehicles queued during the decision cycle, weighted if `use_pcu_weighted_wait=True`:

```python
total_wait = agg.waiting_total(exponent=wait_exponent, use_weights=use_pcu_weighted_wait)
```

(source: [env/sumo_env.py:L660](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L660), [env/mdp_metrics.py:L97-106](file:///c:/Users/Dell/GroupProject2/env/mdp_metrics.py#L97-106))

### Penalty Terms

| Term | Formula | Config Parameter | Source |
|------|---------|------------------|--------|
| **Spillback** | `α * sum(occ²)` | `alpha_spillback: 1.0` | [env/sumo_env.py:L1310-1330](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1310-1330) |

> [!NOTE]
> **Removed components (v2.0):** `lambda_fairness`, `enable_anti_flicker`, `kappa`, `occ_threshold`, `beta`, `teleport_penalty_lambda`, `deadlock_penalty`

### Default Config Values

| Parameter | Value | Source |
|-----------|-------|--------|
| `alpha_spillback` | 1.0 | [configs/train_1.yaml:L59](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L59) |
| `enable_spillback_penalty` | True | [configs/train_1.yaml:L58](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L58) |

### Design Rationale

#### Why This Simplified Formula?

**Academic Foundation:**
- **Varaiya 2013**: Back-pressure control uses queue differentials; squared occupancy provides convex incentive to balance flow
- **PressLight (KDD 2019)**: Uses pressure-based rewards with smooth gradient functions

**Practical Benefits:**
1. **Minimal hyperparameters**: Just 1 parameter (`alpha_spillback`)
2. **Smooth gradients**: Squared occupancy provides gradient from 0%
3. **Pure reward signal**: No auxiliary penalties to confuse learning

#### Hierarchical Penalty Design

| Priority | Component | Purpose |
|----------|-----------|---------|
| **Primary** | `-W/T` | Minimize vehicle delay |
| **Safety** | `-α∑Occ²` | Prevent downstream congestion |

#### Why Spillback is NOT Divided by T?

- **If divided**: At T=60, spillback penalty = 4.0/60 = 0.06 (negligible)
- **Not divided**: Spillback penalty = 4.0 (significant, ~10x waiting penalty)
- **Result**: Agent properly balances throughput vs congestion risk

---

## Training (DQN)

### Algorithm: Double Dueling DQN

**Network Architecture** (Dueling DQN):

```python
class DuelingDQN(nn.Module):
    feature_net = nn.Sequential(
        nn.Linear(state_dim, hidden_1),
        nn.ReLU(),
        nn.Linear(hidden_1, hidden_2),
        nn.ReLU(),
    )
    value_head = nn.Linear(hidden_2, 1)
    advantage_head = nn.Linear(hidden_2, action_dim)
    
    # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
```

(source: [rl/dueling_dqn.py:L9-35](file:///c:/Users/Dell/GroupProject2/rl/dueling_dqn.py#L9-35))

### Update Rule (Double DQN)

```python
# Select best action using online network
next_actions = argmax(online_net(next_states))

# Evaluate using target network
next_q_target = target_net(next_states).gather(1, next_actions)

# TD target
target_q = rewards + gammas * next_q_target * (1 - dones)

# Current Q
current_q = online_net(states).gather(1, actions)

# MSE Loss
loss = MSELoss(current_q, target_q)
```

(source: [rl/agent.py:L143-169](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L143-169))

### Time-Aware Gamma

When `use_time_aware_gamma=True`:

```python
gamma_effective = gamma_0 ** (t_step / t_ref)
```

(source: [rl/agent.py:L86-98](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L86-98))

### Hyperparameters Table

| Parameter | Value | Source |
|-----------|-------|--------|
| `hidden_dims` | [256, 256] | [configs/train_1.yaml:L298](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L298) |
| `gamma` | 0.99 | [configs/train_1.yaml:L299](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L299) |
| `use_time_aware_gamma` | True | [configs/train_1.yaml:L300](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L300) |
| `t_ref` | 60.0 | [configs/train_1.yaml:L301](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L301) |
| `learning_rate` | 0.0001 | [configs/train_1.yaml:L302](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L302) |
| `batch_size` | 256 | [configs/train_1.yaml:L303](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L303) |
| `replay_buffer_size` | 200,000 | [configs/train_1.yaml:L304](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L304) |
| `target_update_freq` | 5000 | [configs/train_1.yaml:L305](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L305) |
| `clip_grad_norm` | 10.0 | [configs/train_1.yaml:L306](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L306) |

### Exploration Schedule

| Parameter | Value | Source |
|-----------|-------|--------|
| `eps_start` | 1.0 | [configs/train_1.yaml:L309](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L309) |
| `eps_end` | 0.02 | [configs/train_1.yaml:L310](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L310) |
| `eps_decay_steps` | 50,000 | [configs/train_1.yaml:L311](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L311) |

Linear decay formula:
```python
epsilon = eps_start + (eps_end - eps_start) * min(1.0, step / eps_decay_steps)
```

(source: [rl/parallel_actors.py:L153-157](file:///c:/Users/Dell/GroupProject2/rl/parallel_actors.py#L153-157))

### Curriculum Learning

| Phase | Episodes | Demand | Route Manifest |
|-------|----------|--------|----------------|
| phase1_warmup | 100 | 50% (400 veh/hr/lane) | `manifest_d400.txt` |
| phase2_moderate | 150 | 75% (600 veh/hr/lane) | `manifest_d600.txt` |
| phase3_baseline | 450 | 100% (800 veh/hr/lane) | `manifest_d800.txt` |
| phase4_high | 200 | 125% (1000 veh/hr/lane) | `manifest_d1000.txt` |
| phase5_stress | 100 | 150% (1200 veh/hr/lane) | `manifest_d1200.txt` |

(source: [configs/train_1.yaml:L321-348](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L321-348))

---

## Normalization

### StateNormalizer

Z-score normalization with clipping:

```python
state_normalized = (state_raw - mean) / (std + eps)
state_normalized = clip(state_normalized, clip_min, clip_max)
```

(source: [env/normalization.py:L15-52](file:///c:/Users/Dell/GroupProject2/env/normalization.py#L15-52))

| Parameter | Value | Source |
|-----------|-------|--------|
| `eps` | 1e-6 | [env/normalization.py:L20](file:///c:/Users/Dell/GroupProject2/env/normalization.py#L20) |
| `clip_min` | -5.0 | [env/normalization.py:L21](file:///c:/Users/Dell/GroupProject2/env/normalization.py#L21) |
| `clip_max` | 5.0 | [env/normalization.py:L22](file:///c:/Users/Dell/GroupProject2/env/normalization.py#L22) |

### Normalization Statistics File

```yaml
normalization:
  file: configs/norm_curriculum_v3.json
```

(source: [configs/train_1.yaml:L294-295](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L294-295))

### Generating Normalization Statistics

Run `collect_norm_curriculum.py` or `collect_norm_parallel.py` to generate real statistics based on simulation data.

(source: [scripts/collect_norm_curriculum.py](file:///c:/Users/Dell/GroupProject2/scripts/collect_norm_curriculum.py), [scripts/collect_norm_parallel.py](file:///c:/Users/Dell/GroupProject2/scripts/collect_norm_parallel.py))

### Action Table Schema Normalization

Legacy `ns_ratio` is automatically converted to `rho_ns`:

```python
if rho_ns is None:
    new_entry["rho_ns"] = ns_ratio
```

(source: [scripts/config_normalization.py:L6-38](file:///c:/Users/Dell/GroupProject2/scripts/config_normalization.py#L6-38))

---

## Parallel Training

### Configuration

```yaml
parallel:
  enabled: true
  num_actors: 2
  base_port: 8813
  base_seed: 42
  chunk_size: 32
  queue_max_chunks: 10
  sync_every_updates: 50
  epsilon_base: 0.2
  epsilon_worker_delta: 0.02
```

(source: [configs/train_parallel_smoke_1.yaml:L176-186](file:///c:/Users/Dell/GroupProject2/configs/train_parallel_smoke_1.yaml#L176-186))

### Port Allocation Rule

Each worker gets a unique SUMO port:

```python
port = base_port + worker_id
```

(source: [env/sumo_env.py:L1565-1567](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1565-1567))

### Per-Worker Epsilon

Workers can have different exploration rates:

```
epsilon_worker = epsilon_base + worker_id * epsilon_worker_delta
```

(source: [configs/train_parallel_smoke_1.yaml:L184-185](file:///c:/Users/Dell/GroupProject2/configs/train_parallel_smoke_1.yaml#L184-185))

### Actor Process

Each actor:
1. Runs an independent SUMO environment
2. Collects transitions and sends them to a shared queue
3. Periodically syncs weights from the learner

(source: [rl/parallel_actors.py:L30-135](file:///c:/Users/Dell/GroupProject2/rl/parallel_actors.py#L30-135))

---

## Setup and Run

### Prerequisites

- Python 3.9+
- SUMO 1.15+ (Eclipse SUMO traffic simulator)
- PyTorch 2.0+

### Installing SUMO

**Windows**:
1. Download from https://sumo.dlr.de/docs/Downloads.php
2. Run installer (e.g., `sumo-win64-X.Y.Z.msi`)
3. Set environment variable: `SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo`
4. Add to PATH: `%SUMO_HOME%\bin`

**Linux (Ubuntu/Debian)**:
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
export SUMO_HOME=/usr/share/sumo
```

### Environment Variables

```powershell
# Windows PowerShell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH += ";$env:SUMO_HOME\bin"
$env:PYTHONPATH = "$PWD"
```

```bash
# Linux/macOS
export SUMO_HOME=/usr/share/sumo
export PATH=$SUMO_HOME/bin:$PATH
export PYTHONPATH=$(pwd)
```

### Install Dependencies

```bash
pip install -r requirements.txt
# or if using pyproject.toml:
pip install -e .
```

### Run Single Training

```bash
python scripts/train.py --config configs/train_1.yaml
```

### Run Evaluation

```bash
python scripts/eval.py --config configs/eval_1.yaml --model models/1/best_model.pt
```

### Run Parallel Training

```bash
python scripts/train_parallel.py --config configs/train_parallel_smoke_1.yaml
```

### Run Tests

```powershell
# Windows PowerShell
$env:PYTHONPATH = "$PWD"
pytest -q
```

```bash
# Linux/macOS
PYTHONPATH=$(pwd) pytest -q
```

### Verification Commands (User-Run)

```powershell
# Check Python version
python --version

# Check sys.path
python -c "import sys; print(sys.path)"

# Check SUMO/TraCI imports
python -c "import traci, sumolib; print('TraCI OK')"

# Run tests
$env:PYTHONPATH="$PWD"; pytest -q
```

---

## Troubleshooting

### Port Collision

**Symptom**: `Address already in use` or `Socket error` during training.

**Solution**:
1. Kill orphan SUMO processes: `taskkill /F /IM sumo.exe` (Windows) or `pkill sumo` (Linux)
2. Change `base_port` in config (e.g., 8900)
3. Use `worker_id` for parallel training to auto-increment ports

(source: [env/sumo_env.py:L1565-1574](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1565-1574))

### SUMO_HOME Missing

**Symptom**: `ImportError: cannot import name 'traci'`

**Solution**:
```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH += ";$env:SUMO_HOME\bin"
```

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'env'`

**Solution**:
```powershell
$env:PYTHONPATH = "$PWD"
```

### DLL Missing (Windows)

**Symptom**: `DLL load failed while importing` errors.

**Solution**:
1. Install Visual C++ Redistributable 2019+
2. Reinstall SUMO
3. Ensure `SUMO_HOME\bin` is in PATH

### Socket Reset by Peer

**Symptom**: `Socket reset by peer` during parallel normalization.

**Solution**:
1. Reduce number of parallel workers
2. Increase wait time between worker launches
3. Check if SUMO is crashing (check route files for valid flows)

---

## Appendix: Sources Inventory

| File | Description |
|------|-------------|
| [env/sumo_env.py](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py) | Main SUMO environment implementation (1645 lines) |
| [env/mdp_metrics.py](file:///c:/Users/Dell/GroupProject2/env/mdp_metrics.py) | Reward computation and cycle metrics aggregator |
| [env/normalization.py](file:///c:/Users/Dell/GroupProject2/env/normalization.py) | StateNormalizer class for z-score normalization |
| [env/kpi.py](file:///c:/Users/Dell/GroupProject2/env/kpi.py) | Episode KPI tracking |
| [rl/agent.py](file:///c:/Users/Dell/GroupProject2/rl/agent.py) | DQNAgent with Double DQN update |
| [rl/dueling_dqn.py](file:///c:/Users/Dell/GroupProject2/rl/dueling_dqn.py) | Dueling DQN network architecture |
| [rl/replay_buffer.py](file:///c:/Users/Dell/GroupProject2/rl/replay_buffer.py) | Experience replay buffer |
| [rl/parallel_actors.py](file:///c:/Users/Dell/GroupProject2/rl/parallel_actors.py) | Parallel training actor processes |
| [scripts/train.py](file:///c:/Users/Dell/GroupProject2/scripts/train.py) | Main training script with curriculum |
| [scripts/eval.py](file:///c:/Users/Dell/GroupProject2/scripts/eval.py) | Evaluation script |
| [scripts/train_parallel.py](file:///c:/Users/Dell/GroupProject2/scripts/train_parallel.py) | Parallel training launcher |
| [scripts/config_normalization.py](file:///c:/Users/Dell/GroupProject2/scripts/config_normalization.py) | Action table schema normalization |
| [configs/train_1.yaml](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml) | Main training config (353 lines) |
| [configs/train_parallel_smoke_1.yaml](file:///c:/Users/Dell/GroupProject2/configs/train_parallel_smoke_1.yaml) | Parallel training smoke test config |
| [configs/norm_curriculum_v3.json](file:///c:/Users/Dell/GroupProject2/configs/norm_curriculum_v3.json) | Normalization statistics |

---

## References and Theoretical Justification

This section provides academic citations and theoretical foundations for the design choices in this project.

### Reward Function Design

**Negative Waiting Time as Reward**

The use of negative cumulative waiting time as the primary reward signal is well-established in traffic signal control literature:

> "Many models define the reward function as the change in cumulative waiting time between adjacent signal cycles. A negative reward is often applied for increased waiting time, encouraging the agent to reduce delays."

**References:**
- [1] Wei, H. et al. (2018). "IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control." KDD 2018. - Uses change in cumulative delay as reward.
- [2] Liang, X. et al. (2019). "Deep Reinforcement Learning for Traffic Light Control in Vehicular Networks." IEEE Transactions on Vehicular Technology. - Weighted combination of queue length and waiting time.
- [3] Zheng, G. et al. (2019). "Learning Phase Competition for Traffic Signal Control." CIKM 2019. - Pressure-based reward as proxy for queue reduction.

**Multi-objective Reward:**

> "Weighted linear combinations allow for balancing various factors, such as minimizing waiting time and queue length, alongside other metrics like delay, travel time, and throughput."

Our reward formula combines waiting time with spillback and teleport penalties, following this multi-objective design pattern.

---

### Deep Q-Network Architecture

**Dueling DQN**

The Dueling DQN architecture separates value and advantage streams for better policy evaluation:

> Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

**References:**
- [4] Wang, Z. et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." ICML 2016. - Original Dueling DQN paper.
- [5] Van Hasselt, H. et al. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI 2016. - Double DQN to reduce overestimation.

**Application to Traffic Signal Control:**
- [6] Genders, W. and Razavi, S. (2019). "An Enhanced Dueling Double Deep Q-Network with Convolutional Block Attention Module for Traffic Signal Optimization." IEEE Access. - D3QN for TSC with SUMO simulation.
- [7] Tan, T. et al. (2019). "Double Deep Q-Network with a Dual-Agent for Traffic Signal Control." MDPI Electronics. - DDQN for four-phase signalized intersections.

---

### Traffic Engineering Standards

**Yellow and All-Red Intervals**

The project uses `yellow_sec=3` and `all_red_sec=2`, consistent with MUTCD guidelines:

> "The MUTCD advises that a yellow change interval should have a minimum duration of 3 seconds and a maximum duration of 6 seconds."

> "The MUTCD recommends that the red clearance interval should not exceed 6 seconds, with some guidelines specifying a minimum duration of 2.0 seconds."

**References:**
- [8] FHWA (2009). "Manual on Uniform Traffic Control Devices (MUTCD)." U.S. Department of Transportation. Section 4D.26.
- [9] ITE (2020). "Traffic Signal Timing Manual." Institute of Transportation Engineers. - ITE kinematic formula for yellow interval.

**Cycle Length Options (60, 90, 120 seconds)**

Standard cycle lengths in urban traffic engineering:

> Typical cycle lengths range from 60-120 seconds for isolated intersections, with shorter cycles (60-90s) preferred for pedestrian-heavy areas and longer cycles (90-120s) for high-volume arterials.

**References:**
- [10] Highway Capacity Manual (HCM) (2016). Transportation Research Board. - Recommends cycle lengths based on intersection geometry and demand.
- [11] Roess, R. P. et al. (2019). "Traffic Engineering." 5th Edition. Pearson. - Standard textbook for signal timing principles.

**Minimum Green Time**

The `g_min_sec=10` ensures pedestrian crossing time and driver expectation:

> Minimum green time should allow pedestrians to enter crosswalk and vehicles to clear queue. Typical values: 7-15 seconds.

---

### State Normalization (Z-Score)

**Importance in Deep RL:**

> "Z-score normalization ensures that all features have comparable scales, preventing features with large numerical values from dominating the learning process. Normalized input data leads to smoother and faster optimization, accelerating the convergence of the training process."

**Formula:** `z = (x - mu) / sigma`

**References:**
- [12] Henderson, P. et al. (2018). "Deep Reinforcement Learning that Matters." AAAI 2018. - Importance of normalization for reproducibility.
- [13] Andrychowicz, M. et al. (2020). "What Matters In On-Policy Reinforcement Learning?" arXiv:2006.05990. - Observation normalization as key factor.

**Clipping to [-5, 5]:**

Prevents extreme outliers from destabilizing training. Common practice in continuous control tasks.

---

### Exploration Strategy (Epsilon-Greedy Decay)

**Linear Decay Schedule:**

> "The seminal DQN paper by Mnih et al. (2015) used a linear decay, annealing epsilon from 1.0 to 0.1 over the first million frames."

Our project uses:
- `eps_start=1.0` (full exploration initially)
- `eps_end=0.02` (2% random actions at convergence)
- `eps_decay_steps=50000` (gradual transition)

**References:**
- [14] Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning." Nature 518. - Original DQN with epsilon decay.
- [15] Schaul, T. et al. (2016). "Prioritized Experience Replay." ICLR 2016. - Alternative exploration via prioritized sampling.

---

### Curriculum Learning

**Progressive Demand Increase:**

> "Curriculum Reinforcement Learning (CRL) aims to improve learning efficiency by structuring a sequence of tasks from easier to more difficult. This mimics how humans learn, by building foundational skills before tackling more complex challenges."

Our curriculum phases (400 -> 600 -> 800 -> 1000 -> 1200 veh/hr/lane) follow this principle.

**References:**
- [16] Bengio, Y. et al. (2009). "Curriculum Learning." ICML 2009. - Foundational paper on curriculum learning.
- [17] Narvekar, S. et al. (2020). "Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey." JMLR. - Comprehensive survey of curriculum RL.

---

### Experience Replay and Target Network

**Replay Buffer:**

> "Experience replay breaks correlations between consecutive samples, improving sample efficiency and stability."

`replay_buffer_size=200000` provides sufficient history for decorrelated sampling.

**Target Network Update:**

> "Target networks stabilize training by providing consistent Q-value targets during updates."

`target_update_freq=5000` balances stability and learning speed.

**References:**
- [18] Mnih, V. et al. (2013). "Playing Atari with Deep Reinforcement Learning." NIPS Workshop. - Introduction of experience replay for DQN.
- [19] Lillicrap, T. P. et al. (2016). "Continuous control with deep reinforcement learning." ICLR 2016. - Soft target updates in DDPG.

---

### Spillback and Gridlock Prevention

**Downstream Occupancy Monitoring:**

> "Spillback occurs when queues extend beyond intersection capacity, blocking upstream traffic. Monitoring downstream occupancy is critical for preventing gridlock."

`occ_threshold=0.65` triggers penalty when downstream edges are 65% occupied.

**References:**
- [20] Varaiya, P. (2013). "Max pressure control of a network of signalized intersections." Transportation Research Part C. - Pressure-based control for network-level coordination.
- [21] Wu, C. et al. (2017). "Flow: Architecture and Benchmarking for Reinforcement Learning in Traffic Control." arXiv:1710.05465. - Benchmark for RL in traffic with SUMO.

---

## Completeness Checklist

| Section | Status | Source Reference |
|---------|--------|------------------|
| **Reward** | Complete | [env/mdp_metrics.py:L135-145](file:///c:/Users/Dell/GroupProject2/env/mdp_metrics.py#L135-145) |
| **State/Observation** | Complete | [env/sumo_env.py:L1074-1086](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1074-1086) |
| **Action Space** | Complete | [env/sumo_env.py:L1509-1563](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1509-1563) |
| **Training (DQN)** | Complete | [rl/agent.py:L143-169](file:///c:/Users/Dell/GroupProject2/rl/agent.py#L143-169) |
| **Normalization** | Complete | [env/normalization.py:L15-52](file:///c:/Users/Dell/GroupProject2/env/normalization.py#L15-52) |
| **Parallel Training** | Complete | [rl/parallel_actors.py:L30-135](file:///c:/Users/Dell/GroupProject2/rl/parallel_actors.py#L30-135) |
| **Setup/Run** | Complete | Multiple config and script files |
| **References** | Complete | Academic citations added |

**All sections populated with sources and academic justifications.**
