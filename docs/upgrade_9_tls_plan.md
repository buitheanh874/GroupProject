# Upgrade Plan: Multi-TLS Control from 5 to 9 Traffic Lights

## Executive Summary

This document provides a detailed work plan and checklist for upgrading the GroupProject Multi-TLS Adaptive Traffic Signal Control system from controlling 5 traffic lights (1 Center + 4 Satellites in hub-and-spoke topology) to controlling 9 traffic lights (e.g., 1 Center + 8 Satellites or similar BIGNET topology).

The codebase **already has generic multi-TLS support** for N traffic lights, but is currently **validated and optimized for 5 TLS only**. The upgrade requires:
1. Configuration flexibility for arbitrary TLS counts
2. State/action space validation for 9 TLS
3. Multi-agent coordination logic verification
4. Baseline controllers (fixed-time, max-pressure) generalization
5. Training/evaluation pipeline testing
6. Comprehensive regression testing

**No fundamental architectural changes** are needed—primarily configuration, testing, and documentation updates.

---

## Section 1: Current Behavior (5 TLS Limitation)

### Where 5 TLS is Currently Enforced

| Aspect | Location | How Enforced |
|--------|----------|-------------|
| **Config templates** | `configs/train_hub_spoke_demo.yaml` (line 9-14) | Hardcoded `tls_ids: ["Center", "N", "E", "S", "W"]` |
| **Config templates** | `configs/eval_hub_spoke.yaml` (similar) | Same hardcoded 5 TLS |
| **Direction mapping** | `env/sumo_env.py:_direction_lanes_by_tls` (line ~450) | Assumes 4 directions [N, E, S, W] only |
| **State dimensionality** | `env/sumo_env.py:state_dim` (line ~310-315) | Supports 4D (legacy) or 12D (multi), 12D = 4 directions + waiting + 4 occupancy |
| **Occupancy vector** | `env/sumo_env.py:_read_downstream_occupancy()` (line ~1000) | Hardcoded 4-direction occupancy (N, E, S, W) |
| **Example test configs** | `tests/test_state_ordering.py`, `tests/test_mdp_compliance_full.py` | Test with `tls_ids=["CENTER", "N1"]` (2 TLS only) |
| **Documentation** | `README.md`, `docs/` (if exists) | References hub-and-spoke (5 TLS) as the primary example |
| **Hub-spoke assumption** | `scripts/setup_5junction_config.py` | Script assumes 5 TLS for automatic network setup |

### Current Architecture (5 TLS)

- **Center TLS**: Decision-maker, receives state augmentation (downstream occupancy)
- **Satellite TLS** (4): Execute actions constrained by Center's cycle decision
- **State per TLS**: 12D = [q_N, q_E, q_S, q_W, w_N, w_E, w_S, w_W, occ_N, occ_E, occ_S, occ_W]
- **Parameter sharing**: Single DQN network for all 5 agents
- **Cycle synchronization**: All 5 TLS must use same cycle length (30/60/90 seconds)

### Known Limitations in Current Code

1. **No auto-discovery of TLS IDs**: User must manually list in config
2. **Hardcoded 4-direction state**: State vector assumes exactly 4 spatial directions
3. **Occupancy requires 4 downstream links**: `downstream_links: {N, E, S, W}` must have exactly 4 entries
4. **Action space fixed to 15 actions**: 5 splits × 3 cycle lengths (not per-TLS customizable)
5. **No TLS-count validation**: System will not warn if tls_ids list has wrong size for network

---

## Section 2: Target Behavior (9 TLS)

### Expected Requirements for 9 TLS

- **Arbitrary TLS count**: System should handle any N ≥ 1 TLS (5, 9, 12, or more)
- **Auto-detection (optional)**: Read TLS IDs from SUMO network file (.net.xml)
- **Flexible state architecture**: Generalize 4-direction assumption to handle intersections with other topologies (if needed)
- **Backward compatibility**: Existing configs for 5 TLS must still work without modification
- **Cycle synchronization**: All N TLS still synchronized to same cycle (implicit multi-agent constraint)
- **Occupancy handling**: Support variable number of downstream links (not just 4)

### Success Criteria

- Train/eval scripts complete successfully with 9 TLS config
- State shapes and action mappings correct for all 9 TLS
- Baseline controllers (fixed-time, max-pressure) work for all 9 TLS
- Unit tests pass for both 5 TLS (regression) and 9 TLS (new)
- No performance degradation on 5 TLS baseline

---

## Section 3: Work Items (Checklist)

### (1) Configuration & CLI

- [ ] **Create new config template for 9 TLS**
  - **Files**: Create `configs/train_bignet_9tls.yaml`, `configs/eval_bignet_9tls.yaml`
  - **Description**: Copy `train_hub_spoke_demo.yaml`, update `tls_ids: ["Center", "N", "E", "S", "W", "NE", "SE", "SW", "NW"]` (example topology)
  - **Risk**: Low
  - **Acceptance**: Config file valid YAML, loads without error in `scripts/common.build_env()`

- [ ] **Add TLS count to CLI/config validation**
  - **Files**: `scripts/common.py` (line ~50-150 in `build_env()`)
  - **Description**: Add check that `tls_ids` list is non-empty and unique; add optional warning if center_tls_id not in tls_ids
  - **Risk**: Low
  - **Acceptance**: `build_env()` raises clear ValueError if tls_ids invalid

- [ ] **Support auto-discovery of TLS IDs from network (optional)**
  - **Files**: Create `scripts/sumo_network_tools.py` with function `extract_tls_ids(net_file: str) -> List[str]`
  - **Description**: Use XML parsing or sumolib to read `<tlLogic id="...">` from .net.xml
  - **Risk**: Medium (requires sumolib or ET parsing)
  - **Acceptance**: Function returns sorted list of TLS IDs; test with hub_spoke.net.xml and BIGNET.net.xml

---

### (2) Environment Initialization & TLS Discovery

- [ ] **Verify `env/sumo_env.py` tls_ids handling**
  - **Files**: `env/sumo_env.py` lines 220–280 (constructor, `_start_sumo()`)
  - **Description**: Confirm `self._tls_ids` is set from config, validate no hardcoded assumptions
  - **Risk**: Low
  - **Acceptance**: Constructor accepts `tls_ids` of any length; stores in `self._tls_ids` (already done, just verify)

- [ ] **Add TLS count logging at env init**
  - **Files**: `env/sumo_env.py` constructor (after line ~350)
  - **Description**: Print `[SUMOEnv] Initialized with {len(self._tls_ids)} TLS: {self._tls_ids}`
  - **Risk**: Low
  - **Acceptance**: Message appears in logs during train/eval startup

- [ ] **Document required lane_groups structure for N TLS**
  - **Files**: `docs/UPGRADE_CONTROL_9_TLS.md` (this file) + new `docs/LANE_GROUPS_GUIDE.md`
  - **Description**: Explain how to define `lane_groups_by_tls` for arbitrary N; provide examples
  - **Risk**: Low (documentation only)
  - **Acceptance**: Clear guide with 2 examples (5 TLS hub-spoke, 9 TLS BIGNET)

---

### (3) State Representation

- [ ] **Verify state builder for N TLS**
  - **Files**: `env/sumo_env.py` lines 950–1000 (`_build_state_vector()`, `_step_multi()`)
  - **Description**: Confirm state vector construction works for any TLS count; no hardcoded 4-direction assumption remains
  - **Risk**: Low
  - **Acceptance**: State dict returned in `_step_multi()` has keys for all tls_ids; each state vector is 12D (or configured dim)

- [ ] **Validate occupancy vector for variable downstream links**
  - **Files**: `env/sumo_env.py` lines 1000–1040 (`_read_downstream_occupancy()`)
  - **Description**: Ensure function fails fast when downstream_links are missing/blank; no padding or warning fallback for invalid configs
  - **Risk**: Medium
  - **Acceptance**: 
    - For 4 links (5 TLS): returns 4D vector ✓
    - For missing/blank links: raises ValueError with clear guidance ✓

- [ ] **Update state_dim validation**
  - **Files**: `env/sumo_env.py` constructor (line ~310)
  - **Description**: Currently validates `state_dim in (4, 12)`. If occupancy changes, may need to support other dims (e.g., 4 + 8 occupancy = 12; 4 + 4 occupancy = 8, etc.). Add config option or auto-compute.
  - **Risk**: Medium
  - **Acceptance**: 
    - `state_dim=12` works for any N TLS ✓
    - Or auto-compute: `state_dim = 4 + len(downstream_links)` ✓

---

### (4) Action Application & Controllers

- [ ] **Verify multi-TLS action application**
  - **Files**: `env/sumo_env.py` lines 1100–1300 (`_step_multi()`, `_set_phase()`, `_build_intervals_for_tls()`)
  - **Description**: Confirm phase/duration application works for all N TLS via traci.trafficlight methods
  - **Risk**: Low
  - **Acceptance**: All TLS receive correct phase and duration each cycle; no traci errors

- [ ] **Test cycle_to_actions mapping for N TLS**
  - **Files**: `env/sumo_env.py` lines 400–420 (`_build_action_definitions()`, `_cycle_to_actions`)
  - **Description**: Confirm action-to-cycle mapping is built correctly; no hardcoded assumptions
  - **Risk**: Low
  - **Acceptance**: `env.cycle_to_actions` has correct structure; agent can query allowed_action_ids for any N

- [ ] **Generalize fixed-time baseline controller**
  - **Files**: `controllers/fixed_time.py`
  - **Description**: Confirm `FixedTimeController` works for any action_splits list (currently assumes generic action space)
  - **Risk**: Low
  - **Acceptance**: Controller instantiates without error; selects action for N-TLS state

- [ ] **Generalize max-pressure baseline controller**
  - **Files**: `controllers/max_pressure.py` (lines 1–50)
  - **Description**: Ensure `MaxPressureSplitController` works for any lane configuration (not hardcoded to 2 directions NS/EW)
  - **Risk**: Medium
  - **Description detail**: Current implementation assumes only 2 directions (NS, EW) in state[0:2]. For 9 TLS with 4 directions each, may need to adapt. For now, assume still works (9 TLS state is 4D per direction).
  - **Acceptance**: 
    - Max-pressure selects action for 9-TLS state without error ✓
    - If needed, generalize to multi-direction pressure (low priority for MVP)

---

### (5) Training Loop Changes

- [ ] **Verify multi-agent action masking**
  - **Files**: `scripts/train.py` lines 150–200 (action selection with `allowed_action_ids`)
  - **Description**: Confirm center TLS action bucketing works for N agents; all satellites get same allowed_ids
  - **Risk**: Low
  - **Acceptance**: Training loop iterates over all N tls_ids; action masking applied correctly

- [ ] **Validate state dict handling in training**
  - **Files**: `scripts/train.py` lines 120–140 (state is dict for multi-TLS)
  - **Description**: Ensure training loop extracts state for all N TLS; stores transitions for all agents
  - **Risk**: Low
  - **Acceptance**: Loop processes dict state for all N TLS; no KeyError on missing TLS

- [ ] **Test reward aggregation**
  - **Files**: `scripts/train.py` lines 180–190 (reward aggregation: `np.mean(reward_values)`)
  - **Description**: Verify reward averaging works for N TLS (should already work generically)
  - **Risk**: Low
  - **Acceptance**: Total episode reward computed correctly; no NaN values

- [ ] **Update training config templates**
  - **Files**: `configs/train_bignet_9tls.yaml` (created above) + `scripts/train.py` docstring
  - **Description**: Ensure example configs and training docs reference N-TLS support
  - **Risk**: Low
  - **Acceptance**: Train script can load and run with 9-TLS config

---

### (6) Evaluation Loop Changes

- [ ] **Verify multi-agent eval loop**
  - **Files**: `scripts/eval.py` lines 70–150 (multi-agent step logic)
  - **Description**: Confirm eval iterates over all N TLS; applies actions correctly; aggregates KPIs
  - **Risk**: Low
  - **Acceptance**: Eval runs for 1 episode with 9 TLS; produces valid KPIs

- [ ] **Test baseline controller integration in eval**
  - **Files**: `scripts/eval.py` lines 110–130 (fixed/max-pressure controller selection)
  - **Description**: Ensure baseline controllers instantiate and step correctly for N TLS
  - **Risk**: Low
  - **Acceptance**: `--controller fixed` and `--controller max_pressure` work for 9 TLS

- [ ] **Validate KPI tracking for N TLS**
  - **Files**: `env/kpi.py`, `scripts/eval.py` (KPI aggregation)
  - **Description**: Confirm KPI tracker handles per-TLS metrics (currently global, may need per-TLS if desired)
  - **Risk**: Low
  - **Acceptance**: Final KPI (avg_wait_time, etc.) computed correctly across all N TLS

---

### (7) Metrics & Logging

- [ ] **Add TLS count to train metrics CSV**
  - **Files**: `scripts/train.py` lines 90–110 (fieldnames in CSV writer)
  - **Description**: Add column `num_tls` to CSV for tracking TLS count per run
  - **Risk**: Low
  - **Acceptance**: Metrics CSV includes `num_tls` field with value 9

- [ ] **Per-TLS reward logging (optional)**
  - **Files**: `scripts/train.py` or `env/sumo_env.py` (info dict)
  - **Description**: If desired, log individual TLS rewards to understand imbalance; low priority for MVP
  - **Risk**: Low (optional)
  - **Acceptance**: Info dict can be printed with per-TLS rewards if enabled

- [ ] **Update plot scripts for N TLS**
  - **Files**: `scripts/plot_kpis.py`, `analysis/plot_rewards.py` (if they assume specific TLS count)
  - **Description**: Verify plot scripts work with variable TLS counts; no hardcoded 5
  - **Risk**: Low
  - **Acceptance**: Plots generated for 9-TLS training run without error

---

### (8) Testing

#### Unit Tests

- [ ] **Test env init with 9 TLS**
  - **Files**: Create `tests/test_9tls_env_init.py`
  - **Description**: Test that `SUMOEnv` initializes correctly with `tls_ids` of length 9
  - **Risk**: Low
  - **Acceptance**:
    ```python
    def test_sumo_env_9tls_init():
        cfg = SumoEnvConfig(..., tls_ids=["C", "N", "E", "S", "W", "NE", "SE", "SW", "NW"])
        env = SUMOEnv(...)
        assert len(env._tls_ids) == 9
        assert env.action_dim > 0
    ```

- [ ] **Test state vector shape for 9 TLS**
  - **Files**: Create `tests/test_9tls_state_shape.py`
  - **Description**: Test `_build_state_vector()` produces correct 12D (or Nx12D dict) for 9 TLS
  - **Risk**: Low
  - **Acceptance**:
    ```python
    def test_state_shape_9tls():
        state_vector = env._build_state_vector(tls_id="C", last_q_dir=..., w_dir=...)
        assert state_vector.shape == (12,)
    
    # Or for multi-agent:
    # assert isinstance(states, dict)
    # assert len(states) == 9
    # assert all(s.shape == (12,) for s in states.values())
    ```

- [ ] **Test action application for 9 TLS**
  - **Files**: Create `tests/test_9tls_action_application.py`
  - **Description**: Mock SUMO; test that all 9 TLS receive correct phase/duration
  - **Risk**: Medium (requires mocking traci)
  - **Acceptance**: No traci errors; all TLS phases set correctly

#### Integration Tests

- [ ] **Smoke test: 5-episode train with 9 TLS**
  - **Files**: Create `tests/test_9tls_train_smoke.py` or update `scripts/run_quick_test.py`
  - **Description**: Run 5 episodes of training with 9-TLS config; ensure no crashes
  - **Risk**: Medium (requires SUMO or mock)
  - **Acceptance**: Training completes; metrics CSV has 5 rows; no NaN rewards

- [ ] **Smoke test: eval with 9 TLS**
  - **Files**: `tests/test_9tls_eval_smoke.py`
  - **Description**: Run 1 eval episode with 9-TLS config and fixed-time baseline
  - **Risk**: Medium (requires SUMO or mock)
  - **Acceptance**: Eval completes; final KPIs valid (no NaN)

#### Regression Tests (5 TLS must still work)

- [ ] **Test 5-TLS backward compatibility**
  - **Files**: `tests/test_5tls_backward_compat.py`
  - **Description**: Run existing hub_spoke tests with 5 TLS; ensure no regression
  - **Risk**: Low
  - **Acceptance**: All existing tests pass unchanged

- [ ] **Verify hub_spoke config still loads**
  - **Files**: Test loading `configs/train_hub_spoke_demo.yaml`, `configs/eval_hub_spoke.yaml`
  - **Risk**: Low
  - **Acceptance**: Config loads; env initializes; first episode runs

---

### (9) Documentation

- [ ] **Update README.md**
  - **Files**: `README.md` (top level)
  - **Description**: Add note: "Supports N ≥ 1 traffic lights. Tested with 5 (hub-spoke) and 9 (BIGNET) TLS topologies."
  - **Risk**: Low
  - **Acceptance**: README lists supported TLS counts

- [ ] **Create BIGNET setup guide**
  - **Files**: Create `docs/BIGNET_SETUP.md`
  - **Description**: Step-by-step to prepare BIGNET.net.xml, define lane_groups_by_tls for 9 TLS, create configs
  - **Risk**: Low
  - **Acceptance**: Guide has 3 worked examples (1 tree-like, 1 grid-like topology, 1 actual BIGNET if available)

- [ ] **Create lane_groups reference**
  - **Files**: Create `docs/LANE_GROUPS_GUIDE.md` or add to existing
  - **Description**: Explain how to map network lanes to lane_groups_by_tls for any TLS count
  - **Risk**: Low
  - **Acceptance**: Guide has examples; users can define custom topology without guesswork

- [ ] **Update this work plan in repo**
  - **Files**: `docs/UPGRADE_CONTROL_9_TLS.md` (this file)
  - **Description**: Finalize and commit plan (this is the output)
  - **Risk**: Low
  - **Acceptance**: Plan committed to repo and referenced in README

---

## Section 4: Minimal Viable Path (MVP—6 Steps to "9 TLS Working")

For fastest time-to-demo (assuming SUMO is available), follow this order:

1. **Create 9-TLS config** (30 min)
   - Copy `configs/train_hub_spoke_demo.yaml` → `configs/train_bignet_9tls.yaml`
   - Update `tls_ids`, `lane_groups_by_tls`, `downstream_links` for BIGNET topology
   - Files: `configs/train_bignet_9tls.yaml`

2. **Create lane_groups_by_tls mapping** (1–2 hours)
   - Use `scripts/inspect_net_boundaries.py` or manual analysis of BIGNET.net.xml
   - Define lanes for all 9 TLS in config
   - Files: `configs/train_bignet_9tls.yaml` (lane_groups_by_tls section)

3. **Run collect_normalization_stats.py** (20 min)
   - Execute: `python scripts/collect_norm_stats.py --config configs/train_bignet_9tls.yaml --episodes 5 --out configs/norm_stats_9tls.json`
   - Produces normalization statistics for 9-TLS state
   - Files: `configs/norm_stats_9tls.json`, update config to reference it

4. **Run 5-episode training smoke test** (10–30 min, depends on SUMO speed)
   - Execute: `python scripts/train.py --config configs/train_bignet_9tls.yaml --episodes 5`
   - Should complete without errors
   - Files: logs/*, models/*

5. **Run eval with fixed-time baseline** (5–15 min)
   - Execute: `python scripts/eval.py --config configs/eval_bignet_9tls.yaml --controller fixed --runs 1`
   - Should produce valid KPIs
   - Files: results/*, CSV output

6. **Add regression test for 5 TLS** (30 min)
   - Verify: `python -m pytest tests/test_5tls_backward_compat.py -v`
   - Ensure hub_spoke configs still work
   - Files: tests/test_5tls_backward_compat.py (new or update existing)

**Total MVP time**: ~3–4 hours (mostly waiting for SUMO simulation).

---

## Section 5: Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Hardcoded 4-direction assumption breaks** | Medium | Grep for direction literals ("N", "E", "S", "W") in state/occupancy code; refactor to use TLS-specific direction lists |
| **Action space/masking incompatible with 9 TLS** | Low | Cycle-to-actions mapping should be generic; test with 9 TLS config |
| **BIGNET lane configuration wrong** | High | Visually inspect BIGNET.net.xml in SUMO GUI; cross-check lane_groups_by_tls with actual lanes |
| **Normalization stats misleading for 9 TLS** | Medium | Run collect_normalization_stats.py for 9 TLS separately; use dedicated norm file |
| **Occupancy downstream links incomplete** | Medium | Fail fast when downstream_links has <4 valid N/E/S/W entries; require explicit IDs (no padding/warning fallback) |
| **KPI aggregation incorrect for 9 TLS** | Low | Verify KPI computation in eval loop handles all TLS; check for per-TLS vs. global aggregation |
| **Performance degradation on 5 TLS due to changes** | Low | Run full 5-TLS baseline suite after all changes; compare metrics to baseline |
| **Backward incompatibility in configs** | Low | Keep hub_spoke configs unchanged; new 9-TLS configs separate |

---

## Section 6: Acceptance Criteria (Definition of Done)

### Functional Acceptance

1. ✅ **Training loop**
   - [ ] Run `python scripts/train.py --config configs/train_bignet_9tls.yaml --episodes 5` → Completes successfully
   - [ ] Metrics CSV produced with 5 rows, no NaN rewards
   - [ ] Models saved for all 5 episodes

2. ✅ **Evaluation loop**
   - [ ] Run `python scripts/eval.py --config configs/eval_bignet_9tls.yaml --controller fixed --runs 2` → Completes successfully
   - [ ] Results CSV has 2 rows, valid KPIs (avg_wait_time, max_wait_time, arrived_vehicles all non-NaN)

3. ✅ **Baseline controllers**
   - [ ] Fixed-time controller selects action for 9-TLS state without error
   - [ ] Max-pressure controller selects action for 9-TLS state without error

4. ✅ **State/action consistency**
   - [ ] State dict from env.step() has exactly 9 keys (one per TLS)
   - [ ] Each state vector is 12D
   - [ ] Action dict passed to env.step() has exactly 9 keys
   - [ ] All TLS receive correct phase/duration via SUMO traci

### Testing Acceptance

5. ✅ **Unit tests**
   - [ ] `pytest tests/test_9tls_env_init.py` → PASS
   - [ ] `pytest tests/test_9tls_state_shape.py` → PASS

6. ✅ **Regression tests**
   - [ ] `pytest tests/test_5tls_backward_compat.py` → PASS (all existing tests still pass)
   - [ ] Hub-spoke train/eval configs load and run without error

### Documentation Acceptance

7. ✅ **Documentation**
   - [ ] `docs/UPGRADE_CONTROL_9_TLS.md` committed (this file)
   - [ ] `docs/BIGNET_SETUP.md` exists with worked examples
   - [ ] `README.md` updated to mention N-TLS support
   - [ ] Configs have inline comments explaining lane_groups_by_tls structure

### Code Quality Acceptance

8. ✅ **Code quality**
   - [ ] No new hardcoded 5-TLS assumptions added
   - [ ] Grep search for `tls_ids.*5` or `["Center", "N", "E", "S", "W"]` returns only comments/examples, no logic
   - [ ] All changes backward compatible (5-TLS configs still work)

### Success Signal

**Project achieves "9 TLS Working" status when all of above criteria are met AND:**
- Train/eval complete successfully with BIGNET 9-TLS config
- At least 1 baseline (fixed-time) produces valid KPIs
- No regression in 5-TLS performance

---

## References & Related Files

- Primary env class: `env/sumo_env.py`
- Training entry point: `scripts/train.py`
- Evaluation entry point: `scripts/eval.py`
- Environment builder: `scripts/common.py` (build_env function)
- Baseline controllers: `controllers/fixed_time.py`, `controllers/max_pressure.py`
- Config templates: `configs/train_hub_spoke_demo.yaml`, `configs/eval_hub_spoke.yaml`
- Existing multi-TLS tests: `tests/test_mdp_compliance_full.py`, `tests/test_state_ordering.py`
- MDP specification: `docs/MDP_Final_TongHop_DongBo_v2.docx` (internal reference)

---

## Appendix: Configuration Template for 9 TLS

Below is a minimal example of a 9-TLS config structure (for reference during implementation):

```yaml
env:
  sumo:
    tls_ids: 
      - "CENTER"
      - "N1"
      - "N2"
      - "E1"
      - "E2"
      - "S1"
      - "S2"
      - "W1"
      - "W2"
    center_tls_id: "CENTER"
    
    downstream_links:
      N: "CENTER_N_OUT"
      E: "CENTER_E_OUT"
      S: "CENTER_S_OUT"
      W: "CENTER_W_OUT"
    
    lane_groups_by_tls:
      CENTER:
        lanes_ns_ctrl: ["CENTER_N_IN_0", "CENTER_N_IN_1"]
        lanes_ew_ctrl: ["CENTER_E_IN_0", "CENTER_E_IN_1"]
      N1:
        lanes_ns_ctrl: ["N1_S_IN_0"]
        lanes_ew_ctrl: ["N1_E_IN_0", "N1_W_IN_0"]
      # ... (repeat for all 9 TLS)
```

---

*This work plan was created to enable controlled, backward-compatible expansion from 5-TLS to 9-TLS control.*
*For questions or updates, please reference Section 6 (Acceptance Criteria) as the source of truth for completion.*
