# GroupProject

Multi-TLS hub-and-spoke setup with shared Dueling DQN. See `docs/README.md` for config keys and `configs/train_hub_spoke_demo.yaml` for the 12D state/action-table sample. Multi-node assets belong under `networks/hub_spoke/`.

- `scenario_calibration` (YAML) auto-injects `env.sumo.vehicle_weights` for PCU-weighted rewards; set `force_calibration_overrides: true` to override user weights, or keep user weights by default.
- Action table schema prefers `rho_ns`; legacy `ns_ratio` is still accepted and normalized to `rho_ns` automatically.
- Route pools support manifests/relative paths; use manifests under `train/` or `eval/` when wiring generated variants.
- Generated Hanoi variants emit SUMO-compatible `flows_*.xml` (vTypes + leaf `<flow>` nodes) and `turns_*.xml` (edgeRelation-only, normalized to 6 decimals).
