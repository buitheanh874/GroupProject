# Evidence Pack Report

**Overview**
- repo: C:\Users\Dell\GroupProject2
- timestamp: 2026-01-19 02:05:00
- commit: a659ac8ea1b9f0758fa7143d1942a61591fcec96

**Gate 1 Feasibility**
- input: evidence_pack/gating_runs.csv (from gating_final_t1000; seeds 42,43; controllers=fixed,max_pressure; horizon=2000,warmup=300)
- metrics (mean±std):
  - d500: fixed/max_pressure comp=1.0000±0 tele=0 no_arrival_max≈13.5 slope=0 → TRAIN_SAFE
  - d750: fixed/max_pressure comp=0.9759±0.0001 tele≈0.0001 no_arrival_max=13 slope≈-46 → TRAIN_SAFE
  - d1000: fixed comp=0.9146 tele=0; max_pressure comp=0.9157 tele≈0.0001 no_arrival_max=5.5 → EVAL_ONLY
- PASS/FAIL @d750: both controllers pass bounded metrics; note only 2 seeds (require ≥3) → Gate1 = PARTIAL
- raw lines: evidence_pack/gating_runs.csv (e.g. rows 1-5)

**Gate 2 Baseline sanity (demand=750, seed=42)**
- input: evidence_pack/eval_results.csv (horizon 1500, warmup 300)
- fixed: wait=740.11s throughput=3.539 comp=0.5503 tele=0
- max_pressure: wait=752.02s throughput=3.475 comp=0.5431 tele=0.000104
- actuated: wait=750.72s throughput=3.558 comp=0.5580 tele=0.000102
- webster: wait=738.30s throughput=3.535 comp=0.5528 tele=0
- gate status: PASS
- raw: first 5 lines of evidence_pack/eval_results.csv

**Gate 2b Baseline audit**
- input: evidence_pack/baseline_audit.txt (200 decisions)
- fixed/max_pressure/actuated/webster: INTERNAL (all within action_id 0..14, cycles {60,90,120}, splits {0.3..0.7})
- gate status: PASS

**Gate 3 Smoke-eval**
- input: evidence_pack/smoke_eval_timeseries.csv (10 entries, demand=750, horizon=750)
- sample lines: see evidence_pack/smoke_eval_proof.txt (episodes 1-10 with avg_wait_time_corr ≈540–598s, completion≈0.28–0.34, tele≈0)
- gate status: PASS

**Gate 4 Curriculum evidence**
- input: logs/gate4_curriculum/gate4_curriculum_hist_curriculum_stats.jsonl → evidence_pack/curriculum_histograms.{md,csv}
- snapshots: ep50/100/150 buffer_hist={"-1":N}, sampled_hist={"-1":256}; single-phase curriculum only, no phase_name mapping (phase idx -1)
- gate status: PARTIAL (histograms present but missing explicit phase mapping/overlap)

**Gate 5 Reproducibility**
- manifest: evidence_pack/manifest.json (seeds 42,43; demands 500/750/1000; route manifests incl. t1000; unseen manifest path)
- missing: episode→route log lines; gate status: PARTIAL

**Final Decision**
- NO-GO (Gate1 lacks ≥3 seeds; Gate4/5 partial; otherwise Gate2/2b/3 pass)
