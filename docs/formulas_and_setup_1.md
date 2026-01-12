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

### Core Formula

```
reward = -(wait_total + fairness_penalty + spill_penalty + anti_flicker_penalty) / t_step
```

(source: [env/mdp_metrics.py:L135-145](file:///c:/Users/Dell/GroupProject2/env/mdp_metrics.py#L135-145))

```python
def compute_normalized_reward(
    wait_total: float,
    t_step: float,
    decision_cycle_sec: float,
    fairness_penalty: float = 0.0,
    spill_penalty: float = 0.0,
    anti_flicker_penalty: float = 0.0,
) -> float:
    denom = float(t_step) if float(t_step) > 0.0 else float(decision_cycle_sec)
    denom = max(1.0, float(denom))
    return -float(wait_total + fairness_penalty + spill_penalty + anti_flicker_penalty) / float(denom)
```

### t_step Calculation

```python
t_step = cycle_sec + 2 * yellow_sec + 2 * all_red_sec
```

(source: [env/sumo_env.py:L664](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L664))

### wait_total Calculation

Accumulated waiting time = sum over all vehicles queued during the decision cycle, weighted if `use_pcu_weighted_wait=True`:

```python
total_wait = agg.waiting_total(exponent=wait_exponent, use_weights=use_pcu_weighted_wait)
```

(source: [env/sumo_env.py:L666](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L666), [env/mdp_metrics.py:L97-106](file:///c:/Users/Dell/GroupProject2/env/mdp_metrics.py#L97-106))

### Penalty Terms

| Term | Formula | Enabled By | Source |
|------|---------|------------|--------|
| **Fairness** | `lambda_fairness * max(wait_per_vehicle)` | `lambda_fairness > 0` | [env/sumo_env.py:L669-674](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L669-674) |
| **Spillback** | `beta * sum(max(occ - occ_threshold, 0))` | `enable_spillback_penalty=True` | [env/sumo_env.py:L1356-1369](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1356-1369) |
| **Anti-Flicker** | `kappa` if `cycle_sec != prev_cycle_sec` else `0` | `enable_anti_flicker=True` | [env/sumo_env.py:L1371-1376](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1371-1376) |
| **Teleport** | `teleport_penalty_lambda * teleport_count` | `teleport_penalty_lambda > 0` | [env/sumo_env.py:L688-690](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L688-690) |

### Reward Time Normalization

When `reward_time_normalize=True`:

```python
reward = reward * t_step / decision_duration_sec
```

(source: [env/sumo_env.py:L695-699](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L695-699))

### Default Config Values

| Parameter | Value | Source |
|-----------|-------|--------|
| `lambda_fairness` | 0.0 | [configs/train_1.yaml:L35](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L35) |
| `enable_spillback_penalty` | True | [configs/train_1.yaml:L58](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L58) |
| `occ_threshold` | 0.65 | [configs/train_1.yaml:L59](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L59) |
| `beta` | 1.0 | [configs/train_1.yaml:L60](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L60) |
| `teleport_penalty_lambda` | 5.0 | [configs/train_1.yaml:L9](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L9) |
| `reward_time_normalize` | True | [configs/train_1.yaml:L45](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L45) |

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

**All sections populated with sources.**
