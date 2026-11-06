# Experiment Memory

- **Primary view**: [`single_stream_stability`](technique-inventory.md) (revision 4)
- **Tasks by tier**: [`EXPERIMENTS_BY_TIER.md`](EXPERIMENTS_BY_TIER.md)
- **Cohort**: device `cuda`, seed 0

## Recorded Candidates

### `online-bispf-frozen` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Freeze the initial prior table to remove sparse-row optimizer drift; all other settings unchanged
- **Parent**: `online-bispf-early`
- **Status**: `FAIL` (`1/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.9`; `modes == 4.0` · finite_template_tv=0.6562, hq=0.3438, live_predictions=480.0000, live_prequential_rmse=0.4052, mean_rmse=0.1570, modes=3.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=0.9375, hq=0.0625, live_predictions=480.0000, live_prequential_rmse=0.5059, mean_rmse=0.2553, modes=1.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **PASS** — terminal 5/5 · direction_acc=1.0000, live_predictions=480.0000, live_prequential_rmse=0.0953, next_frame_rmse=0.0715
  - Tier 1 `seq_img_fork_rollout` (required): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `within_state_modes == 2.0` · branch_tv=0.5000, hq=1.0000, live_predictions=480.0000, live_prequential_rmse=0.1684, mean_rmse=0.0349, within_state_modes=1.0000
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.5319, mean_rmse=0.3484, modes=0.0000
  - Tier 1 `seq_img_rare_mode_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.85`; `rare_mass >= 0.03125`; `support_modes == 4.0` · finite_template_tv=0.6250, hq=0.3750, live_predictions=480.0000, live_prequential_rmse=0.4275, mean_rmse=0.1534, rare_mass=0.0000, support_modes=3.0000
  - Tier 1 `seq_img_delayed_cue` (held): **FAIL** — terminal 0/5 · failed `outcome_accuracy >= 0.9`; `prediction_rmse <= 0.12` · live_predictions=480.0000, live_prequential_rmse=0.1126, outcome_accuracy=0.4844, prediction_rmse=0.2973
  - Tier 1 `seq_img_persistent_fork` (held): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `hq >= 0.85`; `trajectory_consistency >= 0.95`; `within_state_modes == 2.0` · branch_tv=0.7188, depth_1_hq=0.2812, depth_1_modes=1.0000, depth_1_tv=0.7188, depth_2_hq=0.6875, depth_2_modes=1.0000, depth_2_tv=0.5000, depth_3_hq=0.6875, depth_3_modes=1.0000, depth_3_tv=0.5000, hq=0.2812, live_predictions=480.0000, live_prequential_rmse=0.0902, mean_rmse=0.1166, trajectory_consistency=0.2188, within_state_modes=1.0000

### `online-bispf-slow-prior` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Reduce prior learning-rate multiplier from 2 to 0.25 for sparse batch-size-1 row visits
- **Parent**: `online-bispf-early`
- **Status**: `FAIL` (`1/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.9`; `modes == 4.0` · finite_template_tv=0.5938, hq=0.4062, live_predictions=480.0000, live_prequential_rmse=0.3806, mean_rmse=0.1652, modes=3.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.4620, mean_rmse=0.3285, modes=0.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **PASS** — terminal 5/5 · direction_acc=1.0000, live_predictions=480.0000, live_prequential_rmse=0.0795, next_frame_rmse=0.0968
  - Tier 1 `seq_img_fork_rollout` (required): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `within_state_modes == 2.0` · branch_tv=0.5000, hq=1.0000, live_predictions=480.0000, live_prequential_rmse=0.1704, mean_rmse=0.0437, within_state_modes=1.0000
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.4689, mean_rmse=0.3577, modes=0.0000
  - Tier 1 `seq_img_rare_mode_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.85`; `rare_mass >= 0.03125`; `support_modes == 4.0` · finite_template_tv=0.3438, hq=0.6562, live_predictions=480.0000, live_prequential_rmse=0.3824, mean_rmse=0.1239, rare_mass=0.0000, support_modes=3.0000
  - Tier 1 `seq_img_delayed_cue` (held): **FAIL** — terminal 0/5 · failed `outcome_accuracy >= 0.9`; `prediction_rmse <= 0.12` · live_predictions=480.0000, live_prequential_rmse=0.1129, outcome_accuracy=0.5156, prediction_rmse=0.2974
  - Tier 1 `seq_img_persistent_fork` (held): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `hq >= 0.85`; `trajectory_consistency >= 0.95`; `within_state_modes == 2.0` · branch_tv=1.0000, depth_1_hq=0.0000, depth_1_modes=0.0000, depth_1_tv=1.0000, depth_2_hq=0.0938, depth_2_modes=2.0000, depth_2_tv=0.9062, depth_3_hq=0.2812, depth_3_modes=1.0000, depth_3_tv=0.7188, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.1687, mean_rmse=0.1944, trajectory_consistency=0.0000, within_state_modes=0.0000

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

### `online-bcap-adam` (`BCAP`)
- **Changed factors**: ParticleGAN 0.9.6 public bcap_adam trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context
- **Parent**: `stream-bcap-adam`
- **Status**: `FAIL` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.9` · finite_template_tv=0.2500, hq=0.7812, live_predictions=480.0000, live_prequential_rmse=0.3822, mean_rmse=0.0928, modes=4.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.4289, mean_rmse=0.3655, modes=0.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **FAIL** — terminal 0/5 · failed `direction_acc >= 0.9`; `next_frame_rmse <= 0.12` · direction_acc=0.5000, live_predictions=480.0000, live_prequential_rmse=0.1996, next_frame_rmse=0.3388
  - Tier 1 `seq_img_fork_rollout` (required): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `within_state_modes == 2.0` · branch_tv=0.5000, hq=1.0000, live_predictions=480.0000, live_prequential_rmse=0.1678, mean_rmse=0.0485, within_state_modes=1.0000
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.4811, mean_rmse=0.3832, modes=0.0000
  - Tier 1 `seq_img_rare_mode_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.85`; `rare_mass >= 0.03125`; `support_modes == 4.0` · finite_template_tv=0.4688, hq=0.5312, live_predictions=480.0000, live_prequential_rmse=0.4027, mean_rmse=0.1346, rare_mass=0.0312, support_modes=4.0000
  - Tier 1 `seq_img_delayed_cue` (held): **FAIL** — terminal 0/5 · failed `outcome_accuracy >= 0.9`; `prediction_rmse <= 0.12` · live_predictions=480.0000, live_prequential_rmse=0.1105, outcome_accuracy=0.5156, prediction_rmse=0.2986
  - Tier 1 `seq_img_persistent_fork` (held): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `hq >= 0.85`; `trajectory_consistency >= 0.95`; `within_state_modes == 2.0` · branch_tv=0.9688, depth_1_hq=0.4688, depth_1_modes=1.0000, depth_1_tv=0.5312, depth_2_hq=0.5938, depth_2_modes=1.0000, depth_2_tv=0.5000, depth_3_hq=0.0312, depth_3_modes=1.0000, depth_3_tv=0.9688, hq=0.0312, live_predictions=480.0000, live_prequential_rmse=0.1367, mean_rmse=0.1345, trajectory_consistency=0.0000, within_state_modes=1.0000

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

### `online-bispf-early` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context
- **Parent**: `stream-bispf-early`
- **Status**: `FAIL` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **FAIL** — terminal 1/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.9` · finite_template_tv=0.0938, hq=0.9062, live_predictions=480.0000, live_prequential_rmse=0.3770, mean_rmse=0.0778, modes=4.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.4368, mean_rmse=0.3414, modes=0.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **FAIL** — terminal 0/5 · failed `direction_acc >= 0.9`; `next_frame_rmse <= 0.12` · direction_acc=0.9688, live_predictions=480.0000, live_prequential_rmse=0.1400, next_frame_rmse=0.1829
  - Tier 1 `seq_img_fork_rollout` (required): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `hq >= 0.85`; `within_state_modes == 2.0` · branch_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.1655, mean_rmse=0.2615, within_state_modes=0.0000
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.5451, mean_rmse=0.3903, modes=0.0000
  - Tier 1 `seq_img_rare_mode_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.85`; `rare_mass >= 0.03125`; `support_modes == 4.0` · finite_template_tv=0.2812, hq=0.7188, live_predictions=480.0000, live_prequential_rmse=0.4420, mean_rmse=0.1204, rare_mass=0.0000, support_modes=3.0000
  - Tier 1 `seq_img_delayed_cue` (held): **FAIL** — terminal 0/5 · failed `outcome_accuracy >= 0.9`; `prediction_rmse <= 0.12` · live_predictions=480.0000, live_prequential_rmse=0.1147, outcome_accuracy=0.5000, prediction_rmse=0.2987
  - Tier 1 `seq_img_persistent_fork` (held): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `hq >= 0.85`; `trajectory_consistency >= 0.95`; `within_state_modes == 2.0` · branch_tv=1.0000, depth_1_hq=0.9062, depth_1_modes=1.0000, depth_1_tv=0.5000, depth_2_hq=0.6562, depth_2_modes=1.0000, depth_2_tv=0.5000, depth_3_hq=0.0000, depth_3_modes=0.0000, depth_3_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.1646, mean_rmse=0.1402, trajectory_consistency=0.0000, within_state_modes=0.0000

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

### `online-bispf-momentum` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Critic first-moment beta 0.95 carries optimizer evidence across correlated observations; one update per arrival
- **Parent**: `online-bispf-early`
- **Status**: `FAIL` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.9`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.4517, mean_rmse=0.3898, modes=0.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.5128, mean_rmse=0.3233, modes=0.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **FAIL** — terminal 0/5 · failed `direction_acc >= 0.9`; `next_frame_rmse <= 0.12` · direction_acc=0.3438, live_predictions=480.0000, live_prequential_rmse=0.3607, next_frame_rmse=0.4683
  - Tier 1 `seq_img_fork_rollout` (required): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `hq >= 0.85`; `within_state_modes == 2.0` · branch_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.4298, mean_rmse=0.4000, within_state_modes=0.0000
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.5442, mean_rmse=0.4260, modes=0.0000
  - Tier 1 `seq_img_rare_mode_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.85`; `rare_mass >= 0.03125`; `support_modes == 4.0` · finite_template_tv=0.7812, hq=0.2188, live_predictions=480.0000, live_prequential_rmse=0.4548, mean_rmse=0.2145, rare_mass=0.0000, support_modes=1.0000
  - Tier 1 `seq_img_delayed_cue` (held): **FAIL** — terminal 0/5 · failed `outcome_accuracy >= 0.9`; `prediction_rmse <= 0.12` · live_predictions=480.0000, live_prequential_rmse=0.1659, outcome_accuracy=0.5000, prediction_rmse=0.3141
  - Tier 1 `seq_img_persistent_fork` (held): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `hq >= 0.85`; `trajectory_consistency >= 0.95`; `within_state_modes == 2.0` · branch_tv=1.0000, depth_1_hq=0.0000, depth_1_modes=0.0000, depth_1_tv=1.0000, depth_2_hq=0.0000, depth_2_modes=0.0000, depth_2_tv=1.0000, depth_3_hq=0.0000, depth_3_modes=0.0000, depth_3_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.2581, mean_rmse=0.2752, trajectory_consistency=0.0000, within_state_modes=0.0000

### `online-bispf-slow-critic` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context, Reduce critic learning-rate multiplier from 1 to 0.25 to limit chasing one observation
- **Parent**: `online-bispf-early`
- **Status**: `FAIL` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.9`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.5120, mean_rmse=0.3622, modes=0.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=0.9688, hq=0.0312, live_predictions=480.0000, live_prequential_rmse=0.4306, mean_rmse=0.3103, modes=1.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **FAIL** — terminal 0/5 · failed `direction_acc >= 0.9`; `next_frame_rmse <= 0.12` · direction_acc=0.9062, live_predictions=480.0000, live_prequential_rmse=0.2308, next_frame_rmse=0.2079
  - Tier 1 `seq_img_fork_rollout` (required): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `within_state_modes == 2.0` · branch_tv=0.5000, hq=1.0000, live_predictions=480.0000, live_prequential_rmse=0.1768, mean_rmse=0.0535, within_state_modes=1.0000
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.4330, mean_rmse=0.4056, modes=0.0000
  - Tier 1 `seq_img_rare_mode_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.85`; `rare_mass >= 0.03125`; `support_modes == 4.0` · finite_template_tv=0.9062, hq=0.0938, live_predictions=480.0000, live_prequential_rmse=0.4992, mean_rmse=0.3179, rare_mass=0.0000, support_modes=1.0000
  - Tier 1 `seq_img_delayed_cue` (held): **FAIL** — terminal 0/5 · failed `outcome_accuracy >= 0.9`; `prediction_rmse <= 0.12` · live_predictions=480.0000, live_prequential_rmse=0.1331, outcome_accuracy=0.5000, prediction_rmse=0.2956
  - Tier 1 `seq_img_persistent_fork` (held): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `hq >= 0.85`; `trajectory_consistency >= 0.95`; `within_state_modes == 2.0` · branch_tv=1.0000, depth_1_hq=0.5000, depth_1_modes=1.0000, depth_1_tv=0.5000, depth_2_hq=0.0000, depth_2_modes=0.0000, depth_2_tv=1.0000, depth_3_hq=0.0000, depth_3_modes=0.0000, depth_3_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.1912, mean_rmse=0.1597, trajectory_consistency=0.0000, within_state_modes=0.0000

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

### `online-ka2-ready` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Disable auxiliary noise, EMA serving, critic guard and latent damping; retain KA2 critic EMA anchor weight=1, Critic beta2=0.99 (other optimizer betas unchanged) engages KA2 readiness around step 80 within the fixed 480-step budget, Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context
- **Parent**: `stream-ka2-ready`
- **Status**: `FAIL` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.9`; `modes == 4.0` · finite_template_tv=0.3438, hq=0.7188, live_predictions=480.0000, live_prequential_rmse=0.4652, mean_rmse=0.1159, modes=3.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=0.9062, hq=0.0938, live_predictions=480.0000, live_prequential_rmse=0.4884, mean_rmse=0.2511, modes=1.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **FAIL** — terminal 0/5 · failed `next_frame_rmse <= 0.12` · direction_acc=0.9688, live_predictions=480.0000, live_prequential_rmse=0.1704, next_frame_rmse=0.1692
  - Tier 1 `seq_img_fork_rollout` (required): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `hq >= 0.85`; `within_state_modes == 2.0` · branch_tv=1.0000, hq=0.0000, live_predictions=480.0000, live_prequential_rmse=0.1537, mean_rmse=0.2133, within_state_modes=0.0000
  - Tier 1 `seq_img_variable_dwell_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=0.9062, hq=0.0938, live_predictions=480.0000, live_prequential_rmse=0.5029, mean_rmse=0.2531, modes=1.0000
  - Tier 1 `seq_img_rare_mode_bars4` (held): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.85`; `rare_mass >= 0.03125`; `support_modes == 4.0` · finite_template_tv=0.2500, hq=0.7500, live_predictions=480.0000, live_prequential_rmse=0.3053, mean_rmse=0.1065, rare_mass=0.0000, support_modes=3.0000
  - Tier 1 `seq_img_delayed_cue` (held): **FAIL** — terminal 0/5 · failed `outcome_accuracy >= 0.9`; `prediction_rmse <= 0.12` · live_predictions=480.0000, live_prequential_rmse=0.1605, outcome_accuracy=0.5156, prediction_rmse=0.2987
  - Tier 1 `seq_img_persistent_fork` (held): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `hq >= 0.85`; `trajectory_consistency >= 0.95`; `within_state_modes == 2.0` · branch_tv=0.9375, depth_1_hq=0.0625, depth_1_modes=1.0000, depth_1_tv=0.9375, depth_2_hq=0.3125, depth_2_modes=1.0000, depth_2_tv=0.6875, depth_3_hq=0.1562, depth_3_modes=1.0000, depth_3_tv=0.8438, hq=0.0625, live_predictions=480.0000, live_prequential_rmse=0.1862, mean_rmse=0.1971, trajectory_consistency=0.0000, within_state_modes=1.0000

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
