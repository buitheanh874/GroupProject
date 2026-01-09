Executive summary
- Locked SUMO action space to the 5-5-5 mapping with strict validation and canonical ordering for both generated and table-driven actions.
- Standardized phase timeline (EW→yellow→all-red→NS→yellow→all-red) with decision timing `T_step = cycle + 2*yellow + 2*all_red`, and reward time normalization now tied to the measured decision duration.
- Hardened route loading for train/eval with non-empty route checks and manifest-first behavior; refreshed configs and docs to reflect the canonical setup.

Files modified
- README.md
- configs/train_1.yaml
- configs/eval_1.yaml
- configs/eval_bignet_9tls_long.yaml
- env/sumo_env.py
- scripts/train.py
- scripts/eval.py
- tests/test_action_map_1.py
- tests/test_reward_time_1.py

Files removed
- reports/antigravity_run_report.md (obsolete report for 30/60/90 setup; verified no remaining references)

Action mapping (id → cycle, split)
- 0: 60s, (0.30,0.70)
- 1: 60s, (0.40,0.60)
- 2: 60s, (0.50,0.50)
- 3: 60s, (0.60,0.40)
- 4: 60s, (0.70,0.30)
- 5: 90s, (0.30,0.70)
- 6: 90s, (0.40,0.60)
- 7: 90s, (0.50,0.50)
- 8: 90s, (0.60,0.40)
- 9: 90s, (0.70,0.30)
- 10: 120s, (0.30,0.70)
- 11: 120s, (0.40,0.60)
- 12: 120s, (0.50,0.50)
- 13: 120s, (0.60,0.40)
- 14: 120s, (0.70,0.30)

Decision timing and T_step
- Phase order per decision: EW green → EW yellow → all-red → NS green → NS yellow → all-red.
- Decision duration is measured from TraCI before/after the decision loop; reward_time_normalize divides by this measured duration.
- `T_step = cycle_sec + 2*yellow_sec + 2*all_red_sec`; `decision_cycle_sec` now reports the green budget (60/90/120), while `t_step` reports `T_step`.

Manual command checklist (not run)
- pytest -q
- python -m scripts.collect_norm_stats --config configs/train_1.yaml --episodes 50 --out configs/norm_1.json
- python scripts/train.py --config configs/train_1.yaml --episodes 5
- python scripts/eval.py --config configs/eval_1.yaml --controller fixed --runs 1
- python scripts/train.py --config configs/train_1.yaml --episodes 500
