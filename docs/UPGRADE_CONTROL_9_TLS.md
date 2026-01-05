# Upgrade Control to 9 TLS (BIGNET) – Commit-Ready Checklist

This file operationalizes `docs/upgrade_9_tls_plan.md` with concrete code/config/doc artifacts now in the repo.

## What Changed (traceable)
- TLS discovery utility: `scripts/sumo_network_tools.py::extract_tls_ids` (order-preserving, stdlib XML).
- Config validation + auto TLS wiring: `scripts/common.py` (`auto_tls_ids`, uniqueness, center membership, downstream warnings).
- Env robustness: `env/sumo_env.py` (TLS count logging, tolerant downstream occupancy handling).
- Controllers & wiring for N TLS: `controllers/max_pressure.py` (12D-aware + helper), `scripts/train.py`/`scripts/eval.py` (cycle masking helper, multi-TLS baseline actions, metrics `num_tls` column).
- 9-TLS templates: `configs/train_bignet_9tls.yaml`, `configs/eval_bignet_9tls.yaml`, placeholder routes `networks/BIGNET.rou.xml`.
- Tests covering TLS validation/discovery/controllers: `tests/test_tls_discovery.py`.
- README note and run samples updated (see `README.md`).

## Acceptance Criteria (DoD)
- Configs
  - `configs/train_bignet_9tls.yaml` and `configs/eval_bignet_9tls.yaml` load via `scripts/common.build_env` (placeholders are clearly marked).
  - `auto_tls_ids` reads from `networks/BIGNET.net.xml` (9 IDs) with stable ordering.
- Env/Controllers
  - `SUMOEnv` logs TLS count; downstream occupancy missing links → warning + zero-fill (no crash).
  - Baseline controllers return action dicts covering all TLS (fixed/max-pressure) in `scripts/eval.py`.
- Tests
  - Unit tests pass: `tests/test_tls_discovery.py` (TLS validation, auto-discovery, 9-key controller actions).
- Docs
  - This checklist plus README section cites N-TLS support and new configs.

## MVP Run Path (after filling BIGNET lanes/routes)
1) Collect normalization stats (optional but recommended):
   - `python scripts/collect_norm_stats.py --config configs/train_bignet_9tls.yaml --episodes 5 --out configs/norm_stats_bignet_9tls.json`
2) Train smoke (5–10 episodes):
   - `python scripts/train.py --config configs/train_bignet_9tls.yaml --episodes 5`
3) Eval smoke (fixed baseline):
   - `python scripts/eval.py --config configs/eval_bignet_9tls.yaml --controller fixed --runs 1`

## Regression Checklist (5-TLS remains intact)
- Hub-spoke templates untouched: `configs/train_hub_spoke_demo.yaml`, `configs/eval_hub_spoke.yaml`.
- Default behavior preserved: single-TLS configs still work (`tls_id` fallback) and state_dim validation unchanged (4/12).
- Existing tests remain applicable; new tests avoid SUMO/traci dependency.

## Risks & Mitigations
- **Placeholder lanes/routes**: BIGNET templates include `REPLACE_*` lane IDs and a stub `networks/BIGNET.rou.xml`; update before running SUMO.
- **Cycle masking**: Multi-TLS masking now centralised; if a custom controller bypasses `resolve_allowed_action_ids`, ensure it aligns with `env.cycle_to_actions`.
- **Downstream occupancy**: Warnings indicate zero-filled directions; provide `downstream_links` only after confirming center TLS in BIGNET topology.

## Quick File Map
- Plan (reference): `docs/upgrade_9_tls_plan.md`
- Implementation checklist (this file): `docs/UPGRADE_CONTROL_9_TLS.md`
- Code: `scripts/common.py`, `scripts/sumo_network_tools.py`, `env/sumo_env.py`, `controllers/max_pressure.py`, `scripts/train.py`, `scripts/eval.py`
- Configs/Templates: `configs/train_bignet_9tls.yaml`, `configs/eval_bignet_9tls.yaml`, `networks/BIGNET.rou.xml`
- Tests: `tests/test_tls_discovery.py`
