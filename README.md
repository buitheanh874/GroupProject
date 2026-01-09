Action Space 5-5-5
- Cycles: [60, 90, 120]
- Splits (order): (0.30,0.70), (0.40,0.60), (0.50,0.50), (0.60,0.40), (0.70,0.30)
- Action ids: 0-4 = cycle 60, 5-9 = cycle 90, 10-14 = cycle 120

Phase Timeline and Timing
- EW green → EW yellow → all-red → NS green → NS yellow → all-red
- Total decision time T_step = cycle_sec + 2*yellow_sec + 2*all_red_sec
- reward_time_normalize divides reward by the measured decision duration

Configs
- Canonical configs: configs/train_1.yaml, configs/eval_1.yaml, norm stats file configs/norm_1.json
- Route manifests: networks/variants/train/manifest_1.txt, networks/variants/eval/manifest_1.txt
