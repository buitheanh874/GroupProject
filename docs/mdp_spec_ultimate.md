# MDP Specification - Ultimate Scenario (Clean & Realism, Option B)

## Executive Summary

This document specifies the **Ultimate Scenario** MDP for a single 4-way intersection with:
- **Dynamic 4-stage demand** (low → high → low → high)
- **Option B timing**: 60s green budget + 2s yellow transitions (64s wall-clock cycle)
- **All lanes controlled by TLS** (no free-flow right-turn slip lanes)
- **Two fairness variants**: Pure (λ=0.0) and Fair (λ=0.12)

---

## 1. Scope and Key Changes

### 1.1 No Free-Flow Right-Turn
- **All 8 incoming lanes** are controlled by the traffic light system
- **No slip lanes**: All vehicles (straight, left, right) must obey TLS
- This simplifies the MDP by eliminating external flows

### 1.2 Timing Structure (Option B)
- **Green budget per decision step**: 60 seconds (allocated by action split)
- **Yellow transition time**: 2 seconds per phase switch (fixed, not controlled)
- **All-red time**: 0 seconds
- **Wall-clock cycle time**: 64 seconds (60s green + 2×2s yellow)

### 1.3 Episode Design
- **4 stages** × **30 cycles/stage** = **120 cycles total**
- **Stage duration**: 30 × 64s = 1920s
- **Total episode duration**: 7680s (2 hours 8 minutes)
- **Stage boundaries**:
  - Stage 1 (Low): 0–1920s
  - Stage 2 (High): 1920–3840s
  - Stage 3 (Low): 3840–5760s
  - Stage 4 (High): 5760–7680s

---

## 2. Lane Grouping (Simplified)

### 2.1 Controlled Lanes (All Lanes)
```python
LANES_NS_CTRL = [
    "N2TL_0", "N2TL_1", "N2TL_2", "N2TL_3",  # North approach
    "S2TL_0", "S2TL_1", "S2TL_2", "S2TL_3"   # South approach
]

LANES_EW_CTRL = [
    "E2TL_0", "E2TL_1", "E2TL_2", "E2TL_3",  # East approach
    "W2TL_0", "W2TL_1", "W2TL_2", "W2TL_3"   # West approach
]
```

### 2.2 No Slip Lanes
```python
LANES_RT_NS = []  # Empty
LANES_RT_EW = []  # Empty
```

**All vehicles** on all lanes contribute to state and reward.

---

## 3. State Definition

### 3.1 State Vector (4D, Unchanged)
```
s_t = [q_NS, q_EW, w_NS, w_EW]
```

- **q_NS, q_EW**: Queue length (halting vehicles, v < 0.1 m/s) on controlled lanes
- **w_NS, w_EW**: Accumulated waiting time (vehicle-seconds) during the cycle

### 3.2 Normalization (Mandatory)
```python
x_norm = (x - mu_x) / (sigma_x + eps)
s_norm = clip(s_norm, -5, 5)
```

**Normalization stats** must be collected from the same scenario:
```bash
python scripts/collect_norm_stats.py \
  --config configs/train_ultimate_pure.yaml \
  --episodes 5 \
  --seed 0 \
  --out configs/norm_stats_ultimate_clean.json
```

---

## 4. Action Space (Discrete, 5 Choices)

### 4.1 Phase Split Options
| Action ID | (ρ_NS, ρ_EW) | g_NS | g_EW |
|-----------|--------------|------|------|
| 0         | (0.30, 0.70) | 18s  | 42s  |
| 1         | (0.40, 0.60) | 24s  | 36s  |
| 2         | (0.50, 0.50) | 30s  | 30s  |
| 3         | (0.60, 0.40) | 36s  | 24s  |
| 4         | (0.70, 0.30) | 42s  | 18s  |

### 4.2 Minimum Green Constraint
```python
rho_min = 0.1
g_min = 6s
```
Each phase guaranteed **≥6s green** per cycle.

---

## 5. Reward Function

### 5.1 Baseline (Pure)
```python
W_t = w_NS + w_EW
r_t = -W_t / 3600.0
```
**λ = 0.0**: No fairness penalty, pure efficiency optimization.

### 5.2 Fairness Variant (Fair)
```python
W_t = w_NS + w_EW
M_t = max(w_NS, w_EW)
r_t = -(W_t + λ * M_t) / 3600.0
```
**λ = 0.12**: Penalizes worst-case approach delay.

**Note**: Raw reward values are **not directly comparable** between Pure and Fair agents due to different λ.

---

## 6. Phase Program Mapping

```yaml
phase_program:
  ns_green: 0      # Phase index for NS green
  ew_green: 4      # Phase index for EW green
  ns_yellow: 1     # Phase index for NS yellow
  ew_yellow: 5     # Phase index for EW yellow
  all_red: 0       # Unused (all_red_sec = 0)
```

---

## 7. Training Route File

### 7.1 File Reference
```
networks/BI_ultimate_clean.rou.xml
```

### 7.2 Demand Pattern (Conceptual)
```xml
<!-- Stage 1: Low (0-1920s) -->
<flow ... begin="0" end="1920" vehsPerHour="50" />

<!-- Stage 2: High (1920-3840s) -->
<flow ... begin="1920" end="3840" vehsPerHour="150" />

<!-- Stage 3: Low (3840-5760s) -->
<flow ... begin="3840" end="5760" vehsPerHour="50" />

<!-- Stage 4: High (5760-7680s) -->
<flow ... begin="5760" end="7680" vehsPerHour="150" />
```

---

## 8. Key Parameters Summary

| Parameter | Value |
|-----------|------:|
| Green budget per cycle | 60s |
| Yellow per switch | 2s |
| All-red | 0s |
| Wall-clock cycle time | 64s |
| Cycles per stage | 30 |
| Stage duration | 1920s |
| Total episode duration | 7680s |
| max_cycles | 120 |
| max_sim_seconds | 7680 |
| rho_min | 0.1 |
| λ (Pure) | 0.0 |
| λ (Fair) | 0.12 |
| gamma | 0.98 |

---

## 9. Implementation Checklist

- [ ] Route file `BI_ultimate_clean.rou.xml` exists with 4-stage demand
- [ ] All 8 lanes assigned to `lanes_ns_ctrl` or `lanes_ew_ctrl`
- [ ] Slip lane lists are **empty**
- [ ] `yellow_sec = 2`, `all_red_sec = 0`
- [ ] `max_cycles = 120`, `max_sim_seconds = 7680`
- [ ] Normalization stats collected: `configs/norm_stats_ultimate_clean.json`
- [ ] Two training configs: `train_ultimate_pure.yaml` (λ=0.0) and `train_ultimate_fair.yaml` (λ=0.12)

---

## 10. Evaluation Protocol

### 10.1 Battlefield A (In-Distribution)
Evaluate both models on the same Ultimate scenario used for training.

### 10.2 Battlefield B (Out-of-Distribution)
Evaluate on an asymmetric demand route (e.g., `BI_unbalanced.rou.xml`).

### 10.3 Cross-Eval (Comparable Reward)
To compare rewards on the same scale:
- Evaluate **both checkpoints** with λ=0 (Pure config)
- Evaluate **both checkpoints** with λ=0.12 (Fair config)

---

## 11. References

- **Baseline MDP**: `docs/mdp_spec.md`
- **Environment**: `env/sumo_env.py`
- **Training configs**: `configs/train_ultimate_{pure,fair}.yaml`
- **Normalization stats**: `configs/norm_stats_ultimate_clean.json`
- **Training route**: `networks/BI_ultimate_clean.rou.xml`
