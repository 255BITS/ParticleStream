# Experiment Memory

- **Primary view**: [`single_stream_stability`](technique-inventory.md) (revision 4)
- **Tasks by tier**: [`EXPERIMENTS_BY_TIER.md`](EXPERIMENTS_BY_TIER.md)
- **Cohort**: device `cuda`, seed 0

## Recorded Candidates

### `bcap` (`BCAP`)
- **Changed factors**: Stateless BCAP baseline on batch_size=1 sequential stream
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `fast-weight-bcap` (`FastWeight-BCAP`)
- **Changed factors**: Detached FastWeightMemory writer/reader context + bounded 64-frame reservoir replay on batch_size=1 stream
- **Parent**: `bcap`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `fast-weight-bcap-ns` (`Global-Stream-BCAP`)
- **Changed factors**: Non-saturating GAN loss for batch_size=1 stability, Cosine learning rate annealing (lr=0.005, betas=(0.0, 0.999), lr_schedule=cosine, lr_anneal_start=0.5, lr_floor=0.05), 64-frame rehearsal reservoir spanning full dwell=16 4-mode cycle, Temporal velocity direction memory with static noise gating (||diff|| > 0.35), Joint non-linear conditional critic architecture, Dwell-aware rehearsal replay and task-isolated context conditioning
- **Parent**: `fast-weight-bcap`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `ka2` (`KA2`)
- **Changed factors**: KA2 critic EMA anchor on batch_size=1 sequential stream
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-bcap` (`BCAP`)
- **Changed factors**: ParticleGAN 0.9.6 public bcap trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Strict online baseline: explicitly disable every replay path; preserve the complete historical optimization recipe, One chronological observation per update; real/fake batch_size=1; causal serve-before-arrival prediction
- **Parent**: `stream-bcap`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-bcap-adam` (`BCAP`)
- **Changed factors**: ParticleGAN 0.9.6 public bcap_adam trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context
- **Parent**: `stream-bcap-adam`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ

### `online-bcap-adam-causal` (`BCAP`)
- **Changed factors**: ParticleGAN 0.9.6 public bcap_adam trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Causal state reader queries detached associative memory and carries current image, velocity, salient history and bounded repetition age in the existing 16-dimensional context
- **Parent**: `online-bcap-adam`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-bcap-stateless` (`BCAP`)
- **Changed factors**: Stateless BCAP baseline on batch_size=1 sequential stream, Strict online baseline: explicitly disable every replay path; preserve the complete historical optimization recipe, One chronological observation per update; real/fake batch_size=1; causal serve-before-arrival prediction
- **Parent**: `bcap`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ

### `online-bcap-transport` (`BCAP`)
- **Changed factors**: ParticleGAN 0.9.6 public bcap_adam trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Public prior manipulator: uniform critic sampling; generator routes the current observation by sequential clean-center MSE plus 0.01 log row usage, adds weight-10 reconstruction on one generated image, and clears the target before serving
- **Parent**: `online-bcap-adam`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-bcap-transport-causal` (`BCAP`)
- **Changed factors**: ParticleGAN 0.9.6 public bcap_adam trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Causal state reader queries detached associative memory and carries current image, velocity, salient history and bounded repetition age in the existing 16-dimensional context, Public prior manipulator: uniform critic sampling; generator routes the current observation by sequential clean-center MSE plus 0.01 log row usage, adds weight-10 reconstruction on one generated image, and clears the target before serving
- **Parent**: `online-bcap-adam-causal`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-bispf` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Strict online baseline: explicitly disable every replay path; preserve the complete historical optimization recipe, One chronological observation per update; real/fake batch_size=1; causal serve-before-arrival prediction
- **Parent**: `stream-bispf`
- **Status**: `FAIL` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.9` · finite_template_tv=0.1562, hq=0.8750, live_predictions=480.0000, live_prequential_rmse=0.4090, mean_rmse=0.0753, modes=4.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.3602, mean_rmse=0.2743, modes=0.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **FAIL** — terminal 0/5 · failed `next_frame_rmse <= 0.12` · direction_acc=0.9688, live_predictions=480.0000, live_prequential_rmse=0.1549, next_frame_rmse=0.1330
  - Tier 1 `seq_img_fork_rollout` (required): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `within_state_modes == 2.0` · branch_tv=0.5000, hq=1.0000, live_predictions=480.0000, live_prequential_rmse=0.1761, mean_rmse=0.0433, within_state_modes=1.0000
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.4709, mean_rmse=0.2856, modes=0.0000
  - Tier 1 `seq_img_rare_mode_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.85`; `rare_mass >= 0.03125`; `support_modes == 4.0` · finite_template_tv=0.1875, hq=0.8125, live_predictions=480.0000, live_prequential_rmse=0.4194, mean_rmse=0.0966, rare_mass=0.0000, support_modes=3.0000
  - Tier 1 `seq_img_delayed_cue` (held): **FAIL** — terminal 0/5 · failed `outcome_accuracy >= 0.9`; `prediction_rmse <= 0.12` · live_predictions=480.0000, live_prequential_rmse=0.1166, outcome_accuracy=0.5000, prediction_rmse=0.3009
  - Tier 1 `seq_img_persistent_fork` (held): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `hq >= 0.85`; `trajectory_consistency >= 0.95`; `within_state_modes == 2.0` · branch_tv=0.9688, depth_1_hq=0.2500, depth_1_modes=1.0000, depth_1_tv=0.7500, depth_2_hq=0.7500, depth_2_modes=1.0000, depth_2_tv=0.5000, depth_3_hq=0.0312, depth_3_modes=1.0000, depth_3_tv=0.9688, hq=0.0312, live_predictions=480.0000, live_prequential_rmse=0.1802, mean_rmse=0.1508, trajectory_consistency=0.0000, within_state_modes=1.0000

### `online-bispf-early` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context
- **Parent**: `stream-bispf-early`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ

### `online-bispf-early-causal` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Causal state reader queries detached associative memory and carries current image, velocity, salient history and bounded repetition age in the existing 16-dimensional context
- **Parent**: `online-bispf-early`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-bispf-frozen` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Freeze the initial prior table to remove sparse-row optimizer drift; all other settings unchanged
- **Parent**: `online-bispf-early`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ

### `online-bispf-momentum` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Critic first-moment beta 0.95 carries optimizer evidence across correlated observations; one update per arrival
- **Parent**: `online-bispf-early`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ

### `online-bispf-optimized` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf_optimized trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Strict online baseline: explicitly disable every replay path; preserve the complete historical optimization recipe, One chronological observation per update; real/fake batch_size=1; causal serve-before-arrival prediction
- **Parent**: `stream-bispf-optimized`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-bispf-optimized-cool` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf_optimized trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Conservative global DualNorm rates: lr=0.003, D multiplier=1, prior multiplier=2; annealing starts at 0.25, Strict online baseline: explicitly disable every replay path; preserve the complete historical optimization recipe, One chronological observation per update; real/fake batch_size=1; causal serve-before-arrival prediction
- **Parent**: `stream-bispf-optimized-cool`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-bispf-slow-critic` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Reduce critic learning-rate multiplier from 1 to 0.25 to limit chasing one observation
- **Parent**: `online-bispf-early`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ

### `online-bispf-slow-prior` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Reduce prior learning-rate multiplier from 2 to 0.25 for sparse batch-size-1 row visits
- **Parent**: `online-bispf-early`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ

### `online-bispf-transport-causal` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Causal state reader queries detached associative memory and carries current image, velocity, salient history and bounded repetition age in the existing 16-dimensional context, Public prior manipulator: uniform critic sampling; generator routes the current observation by sequential clean-center MSE plus 0.01 log row usage, adds weight-10 reconstruction on one generated image, and clears the target before serving
- **Parent**: `online-bispf-early-causal`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-fast-weight-bcap` (`FastWeight-BCAP`)
- **Changed factors**: Detached FastWeightMemory writer/reader context + bounded 64-frame reservoir replay on batch_size=1 stream, Strict online baseline: explicitly disable every replay path; preserve the complete historical optimization recipe, One chronological observation per update; real/fake batch_size=1; causal serve-before-arrival prediction
- **Parent**: `fast-weight-bcap`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-fast-weight-bcap-ns` (`Global-Stream-BCAP`)
- **Changed factors**: Non-saturating GAN loss for batch_size=1 stability, Cosine learning rate annealing (lr=0.005, betas=(0.0, 0.999), lr_schedule=cosine, lr_anneal_start=0.5, lr_floor=0.05), 64-frame rehearsal reservoir spanning full dwell=16 4-mode cycle, Temporal velocity direction memory with static noise gating (||diff|| > 0.35), Joint non-linear conditional critic architecture, Dwell-aware rehearsal replay and task-isolated context conditioning, Strict online baseline: explicitly disable every replay path; preserve the complete historical optimization recipe, One chronological observation per update; real/fake batch_size=1; causal serve-before-arrival prediction
- **Parent**: `fast-weight-bcap-ns`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ

### `online-ka2` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Strict online baseline: explicitly disable every replay path; preserve the complete historical optimization recipe, One chronological observation per update; real/fake batch_size=1; causal serve-before-arrival prediction
- **Parent**: `stream-ka2`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-ka2-clean` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Disable auxiliary noise, EMA serving, critic guard and latent damping; retain KA2 critic EMA anchor weight=1, Strict online baseline: explicitly disable every replay path; preserve the complete historical optimization recipe, One chronological observation per update; real/fake batch_size=1; causal serve-before-arrival prediction
- **Parent**: `stream-ka2-clean`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-ka2-light-anchor` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Disable auxiliary noise, EMA serving, critic guard and latent damping; retain KA2 critic EMA anchor weight=1, Reduce KA2 EMA anchor weight from 1 to 0.1 globally; retain the KA2 penalty and formulation optimizer, Strict online baseline: explicitly disable every replay path; preserve the complete historical optimization recipe, One chronological observation per update; real/fake batch_size=1; causal serve-before-arrival prediction
- **Parent**: `stream-ka2-light-anchor`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-ka2-ready` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Disable auxiliary noise, EMA serving, critic guard and latent damping; retain KA2 critic EMA anchor weight=1, Critic beta2=0.99 (other optimizer betas unchanged) engages KA2 readiness around step 80 within the fixed 480-step budget, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context
- **Parent**: `stream-ka2-ready`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ

### `online-ka2-ready-causal` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Disable auxiliary noise, EMA serving, critic guard and latent damping; retain KA2 critic EMA anchor weight=1, Critic beta2=0.99 (other optimizer betas unchanged) engages KA2 readiness around step 80 within the fixed 480-step budget, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Causal state reader queries detached associative memory and carries current image, velocity, salient history and bounded repetition age in the existing 16-dimensional context
- **Parent**: `online-ka2-ready`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-ka2-stateless` (`KA2`)
- **Changed factors**: KA2 critic EMA anchor on batch_size=1 sequential stream, Strict online baseline: explicitly disable every replay path; preserve the complete historical optimization recipe, One chronological observation per update; real/fake batch_size=1; causal serve-before-arrival prediction
- **Parent**: `ka2`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `online-ka2-transport-causal` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Disable auxiliary noise, EMA serving, critic guard and latent damping; retain KA2 critic EMA anchor weight=1, Critic beta2=0.99 (other optimizer betas unchanged) engages KA2 readiness around step 80 within the fixed 480-step budget, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Causal state reader queries detached associative memory and carries current image, velocity, salient history and bounded repetition age in the existing 16-dimensional context, Public prior manipulator: uniform critic sampling; generator routes the current observation by sequential clean-center MSE plus 0.01 log row usage, adds weight-10 reconstruction on one generated image, and clears the target before serving
- **Parent**: `online-ka2-ready-causal`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `stream-bcap` (`BCAP`)
- **Changed factors**: ParticleGAN 0.9.6 public bcap trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05)
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `stream-bcap-adam` (`BCAP`)
- **Changed factors**: ParticleGAN 0.9.6 public bcap_adam trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05)
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `stream-bispf` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `stream-bispf-early` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window
- **Parent**: `stream-bispf`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `stream-bispf-optimized` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf_optimized trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `stream-bispf-optimized-cool` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf_optimized trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Conservative global DualNorm rates: lr=0.003, D multiplier=1, prior multiplier=2; annealing starts at 0.25
- **Parent**: `stream-bispf-optimized`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `stream-ka2` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05)
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `stream-ka2-clean` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Disable auxiliary noise, EMA serving, critic guard and latent damping; retain KA2 critic EMA anchor weight=1
- **Parent**: `stream-ka2`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `stream-ka2-light-anchor` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Disable auxiliary noise, EMA serving, critic guard and latent damping; retain KA2 critic EMA anchor weight=1, Reduce KA2 EMA anchor weight from 1 to 0.1 globally; retain the KA2 penalty and formulation optimizer
- **Parent**: `stream-ka2-clean`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt

### `stream-ka2-ready` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Disable auxiliary noise, EMA serving, critic guard and latent damping; retain KA2 critic EMA anchor weight=1, Critic beta2=0.99 (other optimizer betas unchanged) engages KA2 readiness around step 80 within the fixed 480-step budget
- **Parent**: `stream-ka2-clean`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, task_fingerprint differ
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_rare_mode_bars4` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_delayed_cue` (held): **NOT_RUN** — terminal 0/- · reason: no receipt
  - Tier 1 `seq_img_persistent_fork` (held): **NOT_RUN** — terminal 0/- · reason: no receipt


Interpretation and historical evidence: [research notes](research-notes.md).
