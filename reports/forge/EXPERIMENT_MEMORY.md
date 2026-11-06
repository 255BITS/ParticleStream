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

### `online-bcap-adam` (`BCAP`)
- **Changed factors**: ParticleGAN 0.9.6 public bcap_adam trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Explicit replay_policy=none: train every arriving observation once, batch_size=1, Predict before arrival; learn next-frame targets from the preceding detached context
- **Parent**: `stream-bcap-adam`
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
