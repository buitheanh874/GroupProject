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

- **Dimension**: 14D (per TLS agent) - includes local traffic state + global broadcast scalars.

(source: [env/sumo_env.py:L1074-1086](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L1074-1086))

```python
def _build_state_vector(self, tls_id: str, last_q_dir: np.ndarray, w_dir: np.ndarray,
                        n_present_norm: float = 0.0, spill_scalar_norm: float = 0.0) -> np.ndarray:
    occupancy = np.zeros(4, dtype=np.float32)
    if self._enable_downstream_occupancy and tls_id == self._center_tls_id:
        occupancy = self._read_downstream_occupancy()
    
    state = np.zeros(14, dtype=np.float32)     # 14D state (SMDP v5)
    state[0:4] = last_q_dir.astype(np.float32)  # Queue counts per direction
    state[4:8] = w_dir.astype(np.float32)       # Waiting sums per direction
    state[8:12] = occupancy                      # Downstream occupancy per direction
    state[12] = n_present_norm                   # Global vehicle count (broadcast)
    state[13] = spill_scalar_norm                # Global spillback scalar (broadcast)
    return state
```

### Feature Table (14 Dimensions)

| Index | Name | Meaning | Source | Normalization |
|-------|------|---------|--------|---------------|
| 0 | `q_NS_L` | Queue count (NS Left-turn) | Local per-TLS | mean=15, std=12 |
| 1 | `q_NS_T` | Queue count (NS Through) | Local per-TLS | mean=15, std=12 |
| 2 | `q_EW_L` | Queue count (EW Left-turn) | Local per-TLS | mean=15, std=12 |
| 3 | `q_EW_T` | Queue count (EW Through) | Local per-TLS | mean=15, std=12 |
| 4 | `w_NS_L` | Waiting time (NS Left-turn) | Local per-TLS | mean=150, std=120 |
| 5 | `w_NS_T` | Waiting time (NS Through) | Local per-TLS | mean=150, std=120 |
| 6 | `w_EW_L` | Waiting time (EW Left-turn) | Local per-TLS | mean=150, std=120 |
| 7 | `w_EW_T` | Waiting time (EW Through) | Local per-TLS | mean=150, std=120 |
| 8 | `occ_N` | Downstream occupancy (North) | Center TLS only* | mean=0.25, std=0.20 |
| 9 | `occ_E` | Downstream occupancy (East) | Center TLS only* | mean=0.25, std=0.20 |
| 10 | `occ_S` | Downstream occupancy (South) | Center TLS only* | mean=0.25, std=0.20 |
| 11 | `occ_W` | Downstream occupancy (West) | Center TLS only* | mean=0.25, std=0.20 |
| **12** | `n_present_norm` | **Normalized vehicle count** | **Global broadcast** | $\min(1, N/N_{CAP})$, $N_{CAP}=10000$ |
| **13** | `spill_scalar_norm` | **Normalized spillback** | **Global broadcast** | $\min(1, \alpha\sum\text{Occ}^2/(\alpha \cdot M))$ |

*\*Non-center TLS receive zeros for index 8-11.*

> [!IMPORTANT]
> **Index 12-13 (Global Broadcast):** These two dimensions are broadcast to ALL agents to satisfy the Markov property. Without them, the reward depends on unobserved global state (POMDP). Broadcasting ensures full observability (MDP).

(source: [configs/norm_curriculum_v5.json](file:///c:/Users/Dell/GroupProject2/configs/norm_curriculum_v5.json))

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

### Baseline Fairness Definition

**Constraint Parity** (enforced ✅):
- Same action space (15 actions: 3 cycles × 5 splits)
- Same t_step model (cycle + 2×yellow + 2×all_red)
- Same cycle options [60, 90, 120] seconds

**Information Parity** (NOT enforced ❌):
- RL uses 14D normalized state with learned policy
- Max-Pressure uses **raw measurements** (pressure = queue_in - queue_out)
- Fixed-Time uses no state (open-loop)

| Constraint | RL Agent | Fixed-Time | Max-Pressure |
|------------|----------|------------|--------------|
| Cycle options | ✅ [60,90,120]s | ✅ Same | ✅ Same |
| t_step model | ✅ cycle + clearance | ✅ Same | ✅ Same |
| Yellow/all-red | ✅ 3s/2s | ✅ Same | ✅ Same |
| Route pool | ✅ Random from pool | ✅ Same pool | ✅ Same pool |
| State input | 14D normalized | None (open-loop) | **Raw pressure** |
| Decision timing | Per-cycle SMDP | Per-cycle | Per-cycle |

> [!NOTE]
> Fairness is defined as **Constraint Parity**, not Information Parity. This is intentional: RL's advantage should come from *learning* better policies, not from unfair constraints.

---

## Reward Function (SMDP v5 Mainline)

### Core Formula

$$R = -\frac{W_{\text{global}}}{N \cdot t_{\text{ref}}} - \frac{\alpha \sum_{d} \text{Occ}_d^2}{M} \cdot \frac{\Delta t}{t_{\text{ref}}}$$

Where:
- $W_{\text{global}}$ = Total waiting time for **entire network** (sum of all TLS)
- $N$ = `n_present` = Current vehicle count (`traci.vehicle.getIDCount()`)
- $t_{\text{ref}} = 60$ seconds (reference time for scaling)
- $\alpha = 3.0$ (spillback weight, config: `alpha_spillback`)
- $\text{Occ}_d$ = Downstream occupancy for direction $d \in \{N, E, S, W\}$
- $M = 4$ (number of directions)
- $\Delta t$ = `t_step` = cycle_sec + 2×yellow + 2×all_red

(source: [env/mdp_metrics.py:L183-206](file:///c:/Users/Dell/GroupProject2/env/mdp_metrics.py#L183): `compute_normalized_reward_smdp`)

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

### Why This Formula? (SMDP Time-Exposure)

| Design Choice | Rationale |
|---------------|-----------|
| **÷ N** (vehicle count) | Demand-invariant: reward scale doesn't explode with traffic |
| **÷ t_ref** (time normalization) | Prevents "cycle hack": longer cycles don't artificially reduce penalty frequency |
| **Squared spillback** | Convex penalty provides smooth gradient from 0% occupancy |
| **Global reward** | Cooperative MARL: all 9 TLS agents receive same reward |

### Spillback Penalty (Squared Occupancy)

Based on **Varaiya 2013 (Back-Pressure)** and **PressLight (KDD 2019)**:

```python
spill_scalar = alpha * np.sum(downstream_occ ** 2)
spill_term = -(spill_scalar / M) * (t_step / t_ref)
```

| Downstream Occupancy | α∑(Occ)² | Spill Term (Δt=70s) |
|---------------------|----------|---------------------|
| All at 30% | 3.0 × 0.36 = 1.08 | -0.31 |
| All at 50% | 3.0 × 1.0 = 3.0 | -0.88 |
| All at 70% | 3.0 × 1.96 = 5.88 | -1.71 |
| All at 100% | 3.0 × 4.0 = 12.0 | -3.50 |

### Removed Components (Historical/Ablation Only)

> [!NOTE]
> The following were in earlier versions but are **disabled in mainline**:

| Component | Status | Reason |
|-----------|--------|--------|
| Legacy `-W/T` formula | REMOVED | Replaced by SMDP time-exposure |
| Teleport penalty | REMOVED | Simulation artifact, not agent decision |
| Deadlock penalty | REMOVED | Should be prevented by design |
| Anti-flicker | REMOVED | Non-Markovian; squared spillback provides stability |
| Threshold spillback | REMOVED | Hard cutoff gives no gradient below threshold |

### Config Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `alpha_spillback` | 3.0 | [configs/train_1.yaml:L59](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L59) |
| `enable_spillback_penalty` | True | [configs/train_1.yaml:L58](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L58) |
| `t_ref` | 60.0 | Hardcoded in `compute_normalized_reward_smdp` |
| `N_CAP` | 10000 | Hardcoded for state normalization |

### t_step Calculation

```python
t_step = cycle_sec + 2 * yellow_sec + 2 * all_red_sec
# Example: 60 + 2*3 + 2*2 = 70 seconds
```

(source: [env/sumo_env.py:L658](file:///c:/Users/Dell/GroupProject2/env/sumo_env.py#L658))

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
| `eps_start` | 1.0 | [configs/train_1.yaml:L312](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L312) |
| `eps_end` | 0.02 | [configs/train_1.yaml:L313](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L313) |
| `eps_decay_steps` | 180,000 | [configs/train_1.yaml:L314](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L314) |

Linear decay formula:
```python
epsilon = eps_start + (eps_end - eps_start) * min(1.0, step / eps_decay_steps)
```

(source: [rl/parallel_actors.py:L153-157](file:///c:/Users/Dell/GroupProject2/rl/parallel_actors.py#L153-157))

### Curriculum Learning

| Phase | Episodes | Duration | Route Manifest |
|-------|----------|----------|----------------|
| phase1_warmup | 200 | 1200s (20min) | `manifest_d2000.txt` |
| phase2_learn | 400 | 1500s (25min) | `manifest_d2000.txt` |
| phase3_master | 600 | 1800s (30min) | `manifest_d2000.txt` |

**Demand**: 2000 veh/hr/lane (86% motorcycle, 12% car, 2% bus)

**Duration-based curriculum rationale**: Baseline (fixed-time 120s cycle) shows gridlock starts ~800s. Training up to 1800s allows agent to learn pre-gridlock patterns without excessive noise from gridlock states.

(source: [configs/train_1.yaml:L324-346](file:///c:/Users/Dell/GroupProject2/configs/train_1.yaml#L324-346))

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

**Mainline: Duration-Based Curriculum (Fixed Demand)**

> "Curriculum Reinforcement Learning (CRL) aims to improve learning efficiency by structuring a sequence of tasks from easier to more difficult."

Our **mainline curriculum** uses **fixed demand** (800 veh/hr/lane) with **increasing duration**:

| Phase | Episodes | Duration | Demand |
|-------|----------|----------|--------|
| phase1_short | 320 | 1200s (20min) | 800 veh/hr/lane |
| phase2_medium | 400 | 1500s (25min) | 800 veh/hr/lane |
| phase3_full | 480 | 1800s (30min) | 800 veh/hr/lane |

**Rationale**: Gridlock typically starts ~800s with fixed-time baseline. Training with increasing duration allows agent to learn pre-gridlock patterns first.

> [!NOTE]
> **Alternative curriculum (ablation only)**: Progressive demand increase (400→600→800→1000→1200 veh/hr/lane) was explored but is **not mainline**.

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
