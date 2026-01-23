# Curriculum histogram snapshots
- Source: `logs/2/train_v2_optimized_curriculum_stats.jsonl`
- Snapshots (buffer_phase_histogram / sampled_batch_phase_histogram):
  - ep200: buffer {"easy_foundation":25866,"medium_scaleup":9000}, sampled {"easy_foundation":197,"medium_scaleup":59}
  - ep250: buffer {"easy_foundation":25866,"medium_scaleup":18162}, sampled {"easy_foundation":150,"medium_scaleup":106}
  - ep300: buffer {"easy_foundation":25866,"medium_scaleup":27567}, sampled {"easy_foundation":114,"medium_scaleup":142}
- Phase mapping present (phase_idx_to_name), multi-phase coverage; sampled batches include both phases, no extreme skew.
