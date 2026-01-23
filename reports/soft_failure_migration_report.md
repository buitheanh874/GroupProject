# Soft Failure Migration Report

**Date**: 2026-01-09  
**Objective**: Migrate from HARD FAILURE to SOFT FAILURE mode per academic standards (PressLight/IntelliLight)

---

## 1. Executive Summary

System migrated to soft failure mode where:
- Simulation runs full 3600s duration regardless of congestion/deadlock
- Teleports penalized via reward (`λ=5.0`) rather than episode termination
- Agent learns to self-resolve congestion through reward signal
- All deadlock shaping completely disabled (8 parameters)

---

## 2. Files Changed

| File | Change |
|------|--------|
| `configs/train_1.yaml` | Soft failure config, teleport penalty, extended horizon |
| `env/sumo_env.py` | Fixed teleport penalty multi-TLS over-penalization |
| `rl/agent.py` | Added config-driven gradient clipping |
| `scripts/common.py` | Wired `clip_grad_norm` parameter |

---

## 3. Config Changes (Old → New)

### `configs/train_1.yaml`

| Key | Old | New |
|-----|-----|-----|
| `max_sim_seconds` | 1800 | **3600** |
| `teleport_penalty_lambda` | 0.0 | **5.0** |
| `terminate_on_deadlock` | true | **false** |
| `teleport_failure_when_congested` | true | **false** |
| `deadlock_early_no_arrival_sec` | 30.0 | **0** |
| `deadlock_no_arrival_sec` | 150.0 | **0** |
| `deadlock_queue_threshold` | 20.0 | **0** |
| `deadlock_downstream_occ_threshold` | 0.85 | **0** |
| `deadlock_active_min` | 30 | **0** |
| `deadlock_early_penalty_max` | 5.0 | **0** |
| `deadlock_penalty` | 100.0 | **0** |

---

## 4. Environment Audit & Fix Details

### Teleport Penalty Fix (`env/sumo_env.py` L872-877)

**Issue**: Global `decision_teleport_count` applied to each of 9 TLS → 9× over-penalization

**Fix**: Divide penalty by `num_tls`:
```python
num_tls = max(1, len(self._tls_ids))
teleport_penalty = lambda * decision_teleport_count / num_tls
```

### Heuristic Disable Verification

| Heuristic | Code Check | Status |
|-----------|------------|--------|
| `deadlock_no_arrival_sec ≤ 0` | L1193: `if deadlock_limit_sec <= 0.0: return` | ✅ |
| `deadlock_early_no_arrival_sec ≤ 0` | L1197: `if early_limit_sec > 0.0` | ✅ |
| `deadlock_queue_threshold ≤ 0` | L1168: `if queue_thresh <= 0.0` + L1171 | ✅ |

### Termination Check

L668-670 confirms termination uses `max_sim_seconds` with `_stepped_seconds`:
```python
if float(self._stepped_seconds) >= float(self._config.max_sim_seconds):
    done = True
```

---

## 5. Gradient Clipping (`rl/agent.py`)

Added config-driven clipping (default 10.0):
```python
# AgentConfig
clip_grad_norm: Optional[float] = 10.0

# DQNAgent.update()
if self._config.clip_grad_norm is not None and float(self._config.clip_grad_norm) > 0:
    torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=float(self._config.clip_grad_norm))
```

---

## 6. Normalization Update

**Command** (user to run manually):
```bash
python -m scripts.collect_norm_stats --config configs/train_1.yaml --episodes 200 --out configs/norm_soft_1.json
```

After collection, update config:
```yaml
normalization:
  file: configs/norm_soft_1.json
```

---

## 7. QA Instructions

### Epsilon Enforcement

Temporarily modify `exploration` section:
```yaml
exploration:
  eps_start: 1.0
  eps_end: 1.0       # Force random
  eps_decay_steps: 1
```

### Dry-run Command
```bash
python scripts/train.py --config configs/train_1.yaml --episodes 1
```

### Pass Criteria
- `total_stepped_seconds` approaches 3600 in log
- No early termination due to deadlock/teleport
- Reward values present (may be negative)

### After QA
Revert: `eps_end: 0.02`, `eps_decay_steps: 200000`

---

## 8. Test Results

```
tests/test_deadlock_policy.py::test_deadlock_trigger_with_no_arrivals PASSED
tests/test_deadlock_policy.py::test_deadlock_no_trigger_when_low_active PASSED
tests/test_deadlock_policy.py::test_teleport_under_congestion_failure PASSED
tests/test_deadlock_policy.py::test_kpi_deadlock_fields PASSED
tests/test_deadlock_policy.py::test_eval_csv_has_deadlock_columns PASSED
============================== 5 passed ==============================
```

---

## 9. Non-Changes / Guardrails

- ❌ `env/mdp_metrics.py` — NOT modified
- ❌ Network architecture — NOT changed
- ❌ Action space — NOT changed

---

## 10. Teleport Penalty Attribution (Caveat)

Current fix divides global teleport penalty equally across all TLS:
```python
penalty_per_tls = λ * global_teleport_count / num_tls
```

**Trade-off**: This is "system-fair" but does not attribute responsibility to specific TLS causing congestion. For more precise attribution, would need per-zone teleport tracking (not implemented).

**Current approach is acceptable** for training stability and academic standard compliance.

---

## 11. Route Demand Verification ✅

Checked route file: `networks/variants/train/bignet_train_seed00042.rou.xml`

```xml
<flow id="..." begin="0.0" end="3600.0" vehsPerHour="..." />
```

**Result**: All flows span `begin=0.0` to `end=3600.0` — **demand covers full 3600s horizon**. No "empty second half" risk.
