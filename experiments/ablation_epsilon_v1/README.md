# Ablation Epsilon v1 - Experiment Protocol

## Overview

This experiment compares 3 epsilon exploration strategies in parallel DQN training for traffic signal control.

**Research Question:** Does epsilon decay with per-worker diversity improve learning stability and final performance compared to fixed epsilon?

---

## Experimental Design

### Variants

| Variant | Name | Description |
|---------|------|-------------|
| v1 | `fixed` | Fixed ε=0.20 for all workers (baseline) |
| v2 | `decay_nomult` | Decay 0.60→0.05, no worker diversity |
| v3 | `decay_mult` | Decay 0.60→0.05, worker multipliers [0.85, 0.95, 1.05, 1.15] |

### Training Regimen

- **Type:** Curriculum learning
- **Phases:** phase1_d800 (320 eps) → phase2_d800 (400 eps) → phase3_d800 (480 eps)
- **Total:** 1,200 episodes × 4 workers = ~87,000 global decision steps

### Seeds (5 runs per variant)

| Run ID | base_seed | Worker seeds |
|--------|-----------|--------------|
| seed42 | 42 | 42, 10042, 20042, 30042 |
| seed123 | 123 | 123, 10123, 20123, 30123 |
| seed456 | 456 | 456, 10456, 20456, 30456 |
| seed789 | 789 | 789, 10789, 20789, 30789 |
| seed1000 | 1000 | 1000, 11000, 21000, 31000 |

---

## Run Naming Convention

```
run_id = curriculum_seed{seed}

Examples:
- curriculum_seed42
- curriculum_seed123
```

**Full path:**
```
experiments/ablation_epsilon_v1/runs/{variant}/curriculum_seed{seed}/
```

---

## Checkpoint Selection Rule

**PRE-REGISTERED: Final-only**

- Report results using `final.pt` checkpoint
- No cherry-picking based on eval performance
- Rationale: Ensures unbiased comparison; decay should converge naturally

---

## Eval Protocol

### Demands

| Demand | Type | Route File |
|--------|------|------------|
| d400 | In-distribution | networks/variants/eval/bignet_eval_d400.rou.xml |
| d800 | In-distribution (train) | networks/variants/eval/bignet_eval_d800.rou.xml |
| d1000 | In-distribution | networks/variants/eval/bignet_eval_d1000.rou.xml |
| d1200 | OOD Demand | networks/variants/eval/bignet_eval_d1200_ood.rou.xml |

### Eval Seeds

- `eval_seeds = [0, 1, 2, 3, 4]`
- **1 episode per seed per demand** (total: 5 episodes/demand)
- `epsilon_eval = 0.0` (greedy policy)

### Output Files

Each run produces:
```
runs/{variant}/curriculum_seed{seed}/eval/
├── eval_d400.csv       # 5 episodes
├── eval_d800.csv       # 5 episodes
├── eval_d1000.csv      # 5 episodes
└── eval_d1200_ood.csv  # 5 episodes
```

---

## KPIs (Pre-registered)

| KPI | Unit | Primary/Secondary |
|-----|------|-------------------|
| `avg_wait` | seconds | **Primary** |
| `p95_wait` | seconds | Secondary |
| `throughput` | vehicles/sim_hour | Secondary |
| `teleport_rate` | ratio | Secondary (safety) |
| `completion_rate` | ratio | Secondary |

### Convergence Definition

- `avg_wait` improves < 1% for 5 consecutive eval checkpoints
- AND `teleport_rate` does not increase
- Evaluated with ε=0

---

## Baseline Comparison

### Controllers

| Controller | Constraint | Notes |
|------------|------------|-------|
| Fixed-time | Cycle-sync (90s) | All TLS use 90s cycle |
| Max-Pressure | **Cycle-sync** | Constrained to match RL |
| RL (v1/v2/v3) | Cycle-sync | All TLS share cycle |

**IMPORTANT:** Max-Pressure is constrained to cycle-sync (60/90/120s shared) for fair comparison. If unconstrained MP is also reported, label it "MP-unconstrained" and note it's an upper bound, not apples-to-apples.

---

## Required Artifacts per Run

```
runs/{variant}/curriculum_seed{seed}/
├── config_resolved.yaml      # Full config after merge
├── stdout.log                # Training output
├── metrics/
│   ├── train_metrics.csv     # Episode rewards, steps
│   ├── worker_summary.json   # Per-worker stats
│   └── eval_progress.csv     # Periodic eval during training
├── checkpoints/
│   ├── step_XXXXX.pt         # Periodic checkpoints
│   └── final.pt              # Final model
└── eval/
    ├── eval_d400.csv
    ├── eval_d800.csv
    ├── eval_d1000.csv
    └── eval_d1200_ood.csv
```

---

## Summary Artifacts

```
summary/
├── eval_kpi_table.csv           # All runs × demands × KPIs
├── eval_kpi_table_mean_std.csv  # Aggregated by variant × demand
├── learning_curves.png          # Train reward vs global_step
└── ablation_report.md           # 1-page summary
```

---

## Commands

### Training

```bash
# v1: Fixed epsilon
python scripts/train_parallel.py \
  --config experiments/ablation_epsilon_v1/configs/v1_fixed.yaml \
  --seed 42

# v2: Decay, no multiplier
python scripts/train_parallel.py \
  --config experiments/ablation_epsilon_v1/configs/v2_decay_nomult.yaml \
  --seed 42

# v3: Decay + multiplier
python scripts/train_parallel.py \
  --config experiments/ablation_epsilon_v1/configs/v3_decay_mult.yaml \
  --seed 42
```

### Evaluation

```bash
python scripts/eval.py \
  --config experiments/ablation_epsilon_v1/configs/eval.yaml \
  --model experiments/ablation_epsilon_v1/runs/v3_decay_mult/curriculum_seed42/checkpoints/final.pt \
  --demand d800 \
  --eval-seeds 0,1,2,3,4 \
  --output experiments/ablation_epsilon_v1/runs/v3_decay_mult/curriculum_seed42/eval/eval_d800.csv
```

---

## Reproducibility Checklist

- [ ] All runs use same: budget (1200 eps), replay capacity, batch size, optimizer, sync_every_updates
- [ ] Seed protocol documented and worker offsets = base_seed + 10000*worker_id
- [ ] Eval uses fixed route files (no random sampling)
- [ ] Checkpoint rule = final-only (pre-registered)
- [ ] Mean±std computed over ≥5 seeds

---

## Expected Deliverables

1. **3 variants × 5 seeds = 15 training runs**
2. **15 runs × 4 demands = 60 eval result files**
3. **Summary table:** mean±std for each variant × demand
4. **Statistical test:** (optional) paired t-test or Wilcoxon between v1 vs v3
