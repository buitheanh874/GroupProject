# Consistency Checks and TODO List

## C) Consistency/Mismatch List

| Mismatch | Report/Config Says | Code Says | Fix Recommendation |
|----------|-------------------|-----------|-------------------|
| **hidden_dims (eval vs train)** | `eval.yaml:234` says `[192, 192]` | `train_1.yaml:298` says `[256, 256]` | **FIX REQUIRED**: Eval config must match trained model. Update `eval.yaml` line 234 to `[256, 256]` |
| **alpha_spillback (train vs eval)** | `train_1.yaml:60` = 3.0 | `eval.yaml:64` = 1.0 | Intentional difference for eval stability, document in report |
| **fixed_action_id baseline** | `train_1.yaml:339` = 12 | `eval.yaml:238` = 2 | Inconsistent baseline. Verify which action_id=12 or action_id=2 should be used. Action 12 = (120s, 0.50/0.50), Action 2 = (60s, 0.50/0.50) |
| **gamma vs gamma_0** | `train_1.yaml:299` uses `gamma: 0.99` | `agent.py:38` reads `gamma_0` from config with fallback to `gamma` | Works correctly but naming is confusing. Consider renaming to `gamma_0` in config |
| **lane_groups J0 count** | `train_1.yaml:74-94` lists 3 lanes per direction × 4 directions = 12 | `eval.yaml:78-91` lists 2 NS + 6 EW = 8 lanes | **MISMATCH**: Eval config has different lane assignments. Verify correct lanes for evaluation |

## D) TODO List - Missing/Unknown Items

### High Priority
- [ ] **TODO**: RL model evaluation results not found in CSV. Only baselines (fixed, max_pressure, actuated, webster) present in `eval_turn801010_baselines_s10.csv`. Need to run RL model evaluation.
- [ ] **TODO**: Parallel training results/logs not examined. Verify if parallel training (10 workers) produces comparable results to sequential.
- [ ] **TODO**: Unseen demand (1250 veh/hr) evaluation data not in provided CSV. Need separate evaluation run.

### Medium Priority  
- [ ] **TODO**: Training curves/convergence plots not found in repository. Check `logs/` or `results/` directories.
- [ ] **TODO**: Route file validation - verify that route manifests exist for all demand levels.
- [ ] **TODO**: Model checkpoint examination - verify saved model dimensions match config.

### Low Priority / Nice-to-Have
- [ ] **TODO**: Turn ratio distribution (80-10-10 mentioned in filename `train_turn801010`) not documented in code. Add to experimental setup.
- [ ] **TODO**: Network topology diagram not found. Generate from BIGNET.net.xml.
- [ ] **TODO**: Statistical significance tests not implemented. Add confidence intervals to result tables.

### Documentation Gaps
- [ ] **UNKNOWN**: PCU (Passenger Car Unit) weights mentioned in config (`vehicle_weights`) but formula for weighted queue not documented.
- [ ] **UNKNOWN**: Exact teleport_time_cap_sec value used - config shows `null` which defaults to simulation horizon.
- [ ] **UNKNOWN**: Whether downstream_links in center TLS (J0) are lane IDs or edge IDs (code supports both but config uses edge IDs E0-E3).

## E) Verification Status

| Component | Status | Notes |
|-----------|--------|-------|
| State dimension (14D) | ✓ Verified | Matches code and config |
| Action space (15 actions) | ✓ Verified | 3 cycles × 5 splits |
| Reward formula | ✓ Verified | SMDP v5 in mdp_metrics.py |
| DQN architecture | ✓ Verified | Dueling DQN confirmed |
| Double DQN update | ✓ Verified | agent.py:161-165 |
| Time-aware gamma | ✓ Verified | agent.py:88-100 |
| Baselines (4) | ✓ Verified | All controller files present |
| Curriculum (9 phases) | ✓ Verified | train_1.yaml:341-417 |
| KPI corrections | ✓ Verified | kpi.py teleport handling |
