# Curriculum Distribution Update Report

**Date:** 2026-01-14  
**Author:** AI Assistant  

---

## Summary

Removed the 1200 veh/hr stress phase (phase5) from the curriculum and redistributed its 5% allocation to the 1000 veh/hr phase. The curriculum now consists of 4 phases instead of 5.

## Rationale

The user identified that:
1. The 1200 veh/hr (150% demand) phase had only 50 episodes (5% of training)
2. This was insufficient for the model to learn meaningful strategies for extreme congestion
3. The phase risked destabilizing learned behaviors without providing significant benefit
4. Better to allocate those episodes to the 1000 veh/hr phase where more learning can occur

---

## New Curriculum Distribution

### Full Training (1000 episodes) - `train_1.yaml`

| Phase | Demand | Episodes | Percentage | Change |
|-------|--------|----------|------------|--------|
| phase1_warmup | 400 veh/hr (50%) | 100 | 10% | - |
| phase2_moderate | 600 veh/hr (75%) | 150 | 15% | - |
| phase3_baseline | 800 veh/hr (100%) | 500 | 50% | - |
| phase4_high | 1000 veh/hr (125%) | **250** | **25%** | +50 eps (+5%) |
| ~~phase5_stress~~ | ~~1200 veh/hr (150%)~~ | ~~50~~ | ~~5%~~ | **REMOVED** |

**Total:** 1000 episodes

### Short Training (300 episodes) - `train_bignet_short.yaml`

| Phase | Demand | Episodes | Percentage | Change |
|-------|--------|----------|------------|--------|
| phase1_warmup | 400 veh/hr (50%) | 30 | 10% | - |
| phase2_moderate | 600 veh/hr (75%) | 45 | 15% | - |
| phase3_baseline | 800 veh/hr (100%) | 150 | 50% | - |
| phase4_high | 1000 veh/hr (125%) | **75** | **25%** | +15 eps (+5%) |
| ~~phase5_extreme~~ | ~~1200 veh/hr (150%)~~ | ~~15~~ | ~~5%~~ | **REMOVED** |

**Total:** 300 episodes

---

## Files Modified

### Configuration Files

1. **[train_1.yaml](file:///C:/Users/Dell/GroupProject2/configs/train_1.yaml)**
   - Removed `phase5_stress` (1200 veh/hr, 50 episodes)
   - Updated `phase4_high` episodes: 200 → 250
   - Updated descriptions to include percentage allocations

2. **[train_bignet_short.yaml](file:///C:/Users/Dell/GroupProject2/configs/train_bignet_short.yaml)**
   - Removed `phase5_extreme` (1200 veh/hr, 15 episodes)
   - Updated `phase4_high` episodes: 60 → 75
   - Renamed phases for consistency (`phase2_rampup` → `phase2_moderate`, etc.)
   - Updated descriptions to include percentage allocations

### Documentation Files

3. **[hyperparameter_references.md](file:///C:/Users/Dell/GroupProject2/docs/hyperparameter_references.md)**
   - Updated "Your Configuration" section to show 4 phases
   - Updated episode distribution table to reflect new percentages
   - Added note explaining why 1200 veh/hr phase was removed
   - Updated "Variable Episode Length" example YAML
   - Updated "Rationale by Phase" table
   - Updated horizon curriculum pattern description

---

## Visual Comparison

### Before (5 phases)
```
  10%        15%            50%              20%       5%
┌─────┬──────────┬──────────────────────┬───────────┬───┐
│ 400 │   600    │        800           │   1000    │1200│
└─────┴──────────┴──────────────────────┴───────────┴───┘
```

### After (4 phases)
```
  10%        15%            50%                 25%
┌─────┬──────────┬──────────────────────┬───────────────┐
│ 400 │   600    │        800           │     1000      │
└─────┴──────────┴──────────────────────┴───────────────┘
```

---

## Benefits of This Change

1. **More robust high-demand learning**: 25% of episodes (vs 20%) at 1000 veh/hr
2. **Simpler curriculum**: 4 phases easier to analyze and debug
3. **No wasted episodes**: All episodes contribute to meaningful learning
4. **Reduced risk of catastrophic forgetting**: No extreme-demand phase to destabilize learned policies

---

## Training Command

Resume or start training with the updated curriculum:

```bash
# Full training (1000 episodes)
python scripts/train_parallel.py --config configs/train_1.yaml

# Short training (300 episodes)
python scripts/train_parallel.py --config configs/train_bignet_short.yaml
```

---

## Notes

- The normalization files (`norm_curriculum_v4.json`) may need regeneration if they previously included 1200 veh/hr statistics
- Existing checkpoints can still be used for resume (curriculum phase is determined by episode count)
- Manifests for d1200 routes still exist but are no longer referenced by training configs
