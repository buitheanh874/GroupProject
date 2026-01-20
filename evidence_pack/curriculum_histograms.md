# Curriculum histogram snapshots
- Source: `logs/gate4_curriculum/gate4_curriculum_hist_curriculum_stats.jsonl`
- Snapshots (buffer_phase_histogram / sampled_batch_phase_histogram):
  - ep50: buffer {"-1":3879}, sampled {"-1":256}, global_step=431, learner_updates=965
  - ep100: buffer {"-1":8100}, sampled {"-1":256}, global_step=900, learner_updates=2020
  - ep150: buffer {"-1":12852}, sampled {"-1":256}, global_step=1428, learner_updates=3206
- Note: single-phase curriculum; hist index -1 corresponds to default phase (mapping not logged). No skew observed across sampled batches (uniform single-bin).
