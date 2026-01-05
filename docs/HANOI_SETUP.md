# Hanoi Setup (MDP Style)

## Calibration YAML
- Provide `scenario_calibration` pointing to a YAML with `pcu_weights`, `turning.mean_LSR`, and turn mapping (either explicit `turn_mapping` or cardinal helpers `approach_order` + `entry_by_dir` + `exit_by_dir`).
- `apply_calibration_overrides` injects `env.sumo.vehicle_weights` from calibration; set `force_calibration_overrides: true` to override user-specified weights.

## Generating Variants
- `scripts/generate_hanoi_route_variants.py` writes per-variant artifacts under `<out>/<split>/`: `flows_<base>.xml` (vTypes + leaf `<flow>`), `turns_<base>.xml` (edgeRelation-only, 6-decimal normalized), `<base>.rou.xml`, `meta_<base>.json`, and `manifest.txt`.
- Use `--skip-router` to skip jtrrouter/duarouter; meta marks `routed=false` and `.rou.xml` is a placeholder copy of flows.

## Wiring into Train/Eval
- Set `scenario_calibration` in config to auto-inject PCU weights for reward computations.
- Use `<split>.route_pool_manifest` (or route_pool) to point to the generated manifest/rou files; relative paths resolve under project root.

## Troubleshooting
- Missing turn mapping: supply `turn_mapping` or cardinal helpers.
- Router missing: rerun with `--skip-router` or install SUMO routers.
- Manifest path issues: keep manifests under `train/` or `eval/` and use relative paths from project root.
