# Doc Consistency Checklist (Defense Weapon)

> Quick reference for answering hội đồng questions about spec consistency.

**Updated:** 2026-01-16 (SMDP v5 Final - All docs synchronized)

**Status:** ✅ **PASS** - All docs now use same reward/state/curriculum mainline

---

## Spec Matrix

| Component | **Mainline (SMDP v5)** | Ablation Variants | Config Key |
|-----------|------------------------|-------------------|------------|
| **State Dimension** | **14D** | 4D, 12D | `env.sumo.state_dim` |
| **State Index 0-3** | Queue counts (NS_L, NS_T, EW_L, EW_T) | Same | - |
| **State Index 4-7** | Waiting times (NS_L, NS_T, EW_L, EW_T) | Same | - |
| **State Index 8-11** | Downstream occupancy (N, E, S, W) | Zeros for 4D | `env.sumo.enable_downstream_occupancy` |
| **State Index 12** | `n_present_norm` = min(1, N/10000) | N/A for 12D | Hardcoded in `_step_multi` |
| **State Index 13** | `spill_scalar_norm` = normalized spillback | N/A for 12D | Hardcoded in `_step_multi` |
| **Action Space** | 15 = 3 cycles × 5 splits | Same | `cycle_options_sec`, `action_splits` |
| **Reward Formula** | SMDP: $-W/(N \cdot t_{ref}) - \alpha\sum\text{Occ}^2/M \cdot \Delta t/t_{ref}$ | Legacy: $-W/T$ | `compute_normalized_reward_smdp` |
| **Spillback Type** | **Squared**: $\alpha\sum\text{Occ}^2$ | Threshold (historical) | `alpha_spillback` |
| **Anti-flicker** | **DISABLED** | Ablation only | `enable_anti_flicker: false` |
| **Teleport Penalty** | **DISABLED** | Ablation only | `teleport_penalty_lambda: 0` |
| **Deadlock Penalty** | **DISABLED** | Ablation only | `deadlock_penalty: 0` |
| **Discount Factor** | Time-aware: $\gamma_0^{t/t_{ref}}$ | Fixed $\gamma$ | `use_time_aware_gamma` |

---

## Quick Answers for Defense

### Q: "14D nhưng bảng chỉ có 12 dims?"
**A:** Index 12-13 là **global broadcast scalars**:
- Index 12: `n_present_norm` = $\min(1, N/N_{CAP})$ với $N_{CAP}=10000$
- Index 13: `spill_scalar_norm` = normalized spillback scalar

### Q: "Spillback là threshold hay squared?"
**A:** Mainline dùng **squared**: $\alpha\sum\text{Occ}^2$. Threshold-linear là phiên bản cũ, chỉ dùng cho ablation.

### Q: "Anti-flicker là novelty hay đã bỏ?"
**A:** Đã **DISABLED** trong mainline. Lý do: gây dependency non-Markovian. Squared spillback đủ ổn định.

### Q: "Baseline/Max-Pressure dùng state gì?"
**A:** Dùng **14D** (hoặc 12D nếu config), nhưng chỉ center TLS có occupancy thực (8-11). Non-center nhận zeros.

### Q: "Source-of-truth là gì?"
**A:** Code trong `env/sumo_env.py::_build_state_vector` và `env/mdp_metrics.py::compute_normalized_reward_smdp`.

---

## Files Updated for SMDP v5 (14D)

| File | Status | Key Changes |
|------|--------|-------------|
| `env/sumo_env.py` | ✅ Updated | `_build_state_vector` returns 14D, validation allows 14 |
| `env/mdp_metrics.py` | ✅ Updated | Added `compute_normalized_reward_smdp` |
| `configs/train_1.yaml` | ✅ Updated | `state_dim: 14` |
| `configs/eval_1.yaml` | ✅ Updated | `state_dim: 14` |
| `docs/mdp_analysis.md` | ✅ Updated | Complete 14D table, squared spillback |
| `docs/formulas_and_setup_1.md` | ✅ Updated | 14D feature table with index 12-13 |
| `docs/novelty_synthesis.md` | ✅ Updated | Squared spillback, anti-flicker → historical |
| `docs/rl_agent_analysis.md` | ✅ Updated | 14D in table and diagram |
| `docs/MDP_COMPLIANCE.md` | ✅ Updated | Added 14D to variants list |
| `docs/UPGRADE_CONTROL_9_TLS.md` | ✅ Updated | References 12D/14D |

---

## Validation Commands

```bash
# Check config state_dim
grep -r "state_dim" configs/*.yaml

# Check code returns 14D
grep -A5 "_build_state_vector" env/sumo_env.py | grep "np.zeros"

# Run gating tests
python scripts/gating_tests.py

# Pilot run
python scripts/pilot_run_quick.py
```
