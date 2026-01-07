# GroupProject

Multi-TLS hub-and-spoke setup with shared Dueling DQN. See `docs/README.md` for config keys and `configs/train_hub_spoke_demo.yaml` for the 12D state/action-table sample. Multi-node assets belong under `networks/hub_spoke/`.

- `scenario_calibration` (YAML) auto-injects `env.sumo.vehicle_weights` for PCU-weighted rewards; set `force_calibration_overrides: true` to override user weights, or keep user weights by default.
- Action table schema prefers `rho_ns`; legacy `ns_ratio` is still accepted and normalized to `rho_ns` automatically.
- Route pools support manifests/relative paths; use manifests under `train/` or `eval/` when wiring generated variants.
- Generated Hanoi variants emit SUMO-compatible `flows_*.xml` (vTypes + leaf `<flow>` nodes) and `turns_*.xml` (edgeRelation-only, normalized to 6 decimals).
- See `docs/HANOI_SETUP.md` for calibration/variant wiring and `docs/MDP_COMPLIANCE.md` for MDP compliance notes.

## Multi-TLS Coverage
- Supports arbitrary `tls_ids` (N TLS). Verified with hub-spoke (5 TLS) and BIGNET template (9 TLS).
- New templates: `configs/train_bignet_9tls.yaml`, `configs/eval_bignet_9tls.yaml` (use `auto_tls_ids: true` to discover IDs from `networks/BIGNET.net.xml`).
- Quick smoke commands: `python scripts/train.py --config configs/train_bignet_9tls.yaml --episodes 5` and `python scripts/eval.py --config configs/eval_bignet_9tls.yaml --controller fixed --runs 1` (fill route/lane mappings first).
