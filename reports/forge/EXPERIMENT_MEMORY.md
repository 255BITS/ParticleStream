# Experiment Memory

- **Primary view**: [`single_stream_stability`](technique-inventory.md) (revision 2)
- **Tasks by tier**: [`EXPERIMENTS_BY_TIER.md`](EXPERIMENTS_BY_TIER.md)
- **Cohort**: device `cuda`, seed 0

## Recorded Candidates

### `stream-bcap-adam` (`BCAP`)
- **Changed factors**: ParticleGAN 0.9.6 public bcap_adam trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05)
- **Status**: `PASS` (`4/4` required passes, qualified tier 1)
  - Tier 1 `seq_img_round_robin_bars4` (required): **PASS** — terminal 5/5 · finite_template_tv=0.0625, hq=1.0000, mean_rmse=0.0276, modes=4.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **PASS** — terminal 5/5 · finite_template_tv=0.0625, hq=1.0000, mean_rmse=0.0305, modes=4.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **PASS** — terminal 5/5 · direction_acc=1.0000, next_frame_rmse=0.0822
  - Tier 1 `seq_img_fork_rollout` (required): **PASS** — terminal 5/5 · branch_tv=0.0625, hq=1.0000, mean_rmse=0.0533, within_state_modes=2.0000

### `stream-bispf-early` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Anneal from step 120 (0.25 of budget) to settle before the sustained terminal window
- **Parent**: `stream-bispf`
- **Status**: `PASS` (`4/4` required passes, qualified tier 1)
  - Tier 1 `seq_img_round_robin_bars4` (required): **PASS** — terminal 5/5 · finite_template_tv=0.0938, hq=0.9688, mean_rmse=0.0395, modes=4.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **PASS** — terminal 5/5 · finite_template_tv=0.0625, hq=0.9688, mean_rmse=0.0347, modes=4.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **PASS** — terminal 5/5 · direction_acc=0.9688, next_frame_rmse=0.0896
  - Tier 1 `seq_img_fork_rollout` (required): **PASS** — terminal 5/5 · branch_tv=0.1250, hq=0.8750, mean_rmse=0.0614, within_state_modes=2.0000

### `stream-ka2-ready` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Disable auxiliary noise, EMA serving, critic guard and latent damping; retain KA2 critic EMA anchor weight=1, Critic beta2=0.99 (other optimizer betas unchanged) engages KA2 readiness around step 80 within the fixed 480-step budget
- **Parent**: `stream-ka2-clean`
- **Status**: `PASS` (`4/4` required passes, qualified tier 1)
  - Tier 1 `seq_img_round_robin_bars4` (required): **PASS** — terminal 5/5 · finite_template_tv=0.0625, hq=1.0000, mean_rmse=0.0287, modes=4.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **PASS** — terminal 5/5 · finite_template_tv=0.0625, hq=1.0000, mean_rmse=0.0288, modes=4.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **PASS** — terminal 5/5 · direction_acc=1.0000, next_frame_rmse=0.0887
  - Tier 1 `seq_img_fork_rollout` (required): **PASS** — terminal 5/5 · branch_tv=0.0938, hq=0.9062, mean_rmse=0.0591, within_state_modes=2.0000

### `stream-ka2` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05)
- **Status**: `FAIL` (`3/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **PASS** — terminal 5/5 · finite_template_tv=0.0938, hq=1.0000, mean_rmse=0.0266, modes=4.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **PASS** — terminal 5/5 · finite_template_tv=0.0312, hq=1.0000, mean_rmse=0.0277, modes=4.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **FAIL** — terminal 0/5 · failed `direction_acc >= 0.9`; `next_frame_rmse <= 0.12` · direction_acc=0.5000, next_frame_rmse=0.3173
  - Tier 1 `seq_img_fork_rollout` (required): **PASS** — terminal 5/5 · branch_tv=0.0938, hq=0.9062, mean_rmse=0.0539, within_state_modes=2.0000

### `stream-ka2-clean` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Disable auxiliary noise, EMA serving, critic guard and latent damping; retain KA2 critic EMA anchor weight=1
- **Parent**: `stream-ka2`
- **Status**: `FAIL` (`3/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **PASS** — terminal 5/5 · finite_template_tv=0.0312, hq=1.0000, mean_rmse=0.0241, modes=4.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **PASS** — terminal 5/5 · finite_template_tv=0.0312, hq=1.0000, mean_rmse=0.0324, modes=4.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **FAIL** — terminal 0/5 · failed `direction_acc >= 0.9`; `next_frame_rmse <= 0.12` · direction_acc=0.4844, next_frame_rmse=0.4160
  - Tier 1 `seq_img_fork_rollout` (required): **PASS** — terminal 5/5 · branch_tv=0.1250, hq=0.9062, mean_rmse=0.0567, within_state_modes=2.0000

### `stream-ka2-light-anchor` (`KA2`)
- **Changed factors**: ParticleGAN 0.9.6 public ka2 trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), Disable auxiliary noise, EMA serving, critic guard and latent damping; retain KA2 critic EMA anchor weight=1, Reduce KA2 EMA anchor weight from 1 to 0.1 globally; retain the KA2 penalty and formulation optimizer
- **Parent**: `stream-ka2-clean`
- **Status**: `FAIL` (`3/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **PASS** — terminal 5/5 · finite_template_tv=0.0312, hq=1.0000, mean_rmse=0.0241, modes=4.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **PASS** — terminal 5/5 · finite_template_tv=0.0312, hq=1.0000, mean_rmse=0.0324, modes=4.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **FAIL** — terminal 0/5 · failed `direction_acc >= 0.9`; `next_frame_rmse <= 0.12` · direction_acc=0.4844, next_frame_rmse=0.4160
  - Tier 1 `seq_img_fork_rollout` (required): **PASS** — terminal 5/5 · branch_tv=0.1250, hq=0.9062, mean_rmse=0.0567, within_state_modes=2.0000

### `stream-bispf` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training
- **Status**: `FAIL` (`2/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **PASS** — terminal 5/5 · finite_template_tv=0.0625, hq=1.0000, mean_rmse=0.0270, modes=4.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **PASS** — terminal 5/5 · finite_template_tv=0.0312, hq=1.0000, mean_rmse=0.0261, modes=4.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **FAIL** — terminal 4/5 · failed `next_frame_rmse <= 0.12` · direction_acc=0.9844, next_frame_rmse=0.0806
  - Tier 1 `seq_img_fork_rollout` (required): **FAIL** — terminal 4/5 · failed `hq >= 0.85` · branch_tv=0.0938, hq=0.9062, mean_rmse=0.0597, within_state_modes=2.0000

### `bcap` (`BCAP`)
- **Changed factors**: Stateless BCAP baseline on batch_size=1 sequential stream
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ

### `fast-weight-bcap` (`FastWeight-BCAP`)
- **Changed factors**: Detached FastWeightMemory writer/reader context + bounded 64-frame reservoir replay on batch_size=1 stream
- **Parent**: `bcap`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ

### `fast-weight-bcap-ns` (`Global-Stream-BCAP`)
- **Changed factors**: Non-saturating GAN loss for batch_size=1 stability, Cosine learning rate annealing (lr=0.005, betas=(0.0, 0.999), lr_schedule=cosine, lr_anneal_start=0.5, lr_floor=0.05), 64-frame rehearsal reservoir spanning full dwell=16 4-mode cycle, Temporal velocity direction memory with static noise gating (||diff|| > 0.35), Joint non-linear conditional critic architecture, Dwell-aware rehearsal replay and task-isolated context conditioning
- **Parent**: `fast-weight-bcap`
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ

### `ka2` (`KA2`)
- **Changed factors**: KA2 critic EMA anchor on batch_size=1 sequential stream
- **Status**: `INCOMPLETE` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_dwell_bars4` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_translating_bar_next` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ
  - Tier 1 `seq_img_fork_rollout` (required): **NOT_RUN** — terminal 0/- · reason: stale receipt: adapter_source_sha256, candidate_fingerprint, device, engine_source_sha256, namespace, rng_version, seed, task_fingerprint, cohort_key differ

### `stream-bcap` (`BCAP`)
- **Changed factors**: ParticleGAN 0.9.6 public bcap trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05)
- **Status**: `FAIL` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.9`; `modes == 4.0` · finite_template_tv=0.9688, hq=0.0312, mean_rmse=0.2082, modes=1.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=0.6250, hq=0.4375, mean_rmse=0.1424, modes=2.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **FAIL** — terminal 0/5 · failed `direction_acc >= 0.9`; `next_frame_rmse <= 0.12` · direction_acc=0.5000, next_frame_rmse=0.3576
  - Tier 1 `seq_img_fork_rollout` (required): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `hq >= 0.85`; `within_state_modes == 2.0` · branch_tv=0.5000, hq=0.6250, mean_rmse=0.1355, within_state_modes=1.0000

### `stream-bispf-optimized` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf_optimized trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training
- **Status**: `FAIL` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.9`; `modes == 4.0` · finite_template_tv=0.8438, hq=0.1562, mean_rmse=0.1824, modes=3.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=0.5938, hq=0.5312, mean_rmse=0.1423, modes=1.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **FAIL** — terminal 0/5 · failed `direction_acc >= 0.9`; `next_frame_rmse <= 0.12` · direction_acc=0.5000, next_frame_rmse=0.3664
  - Tier 1 `seq_img_fork_rollout` (required): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `hq >= 0.85`; `within_state_modes == 2.0` · branch_tv=0.5000, hq=0.5312, mean_rmse=0.1416, within_state_modes=1.0000

### `stream-bispf-optimized-cool` (`BISPF`)
- **Changed factors**: ParticleGAN 0.9.6 public bispf_optimized trainer preset, Shared 64-frame FIFO memory, seven replay samples and velocity context, Global non-saturating loss and cosine schedule (start=0.5, floor=0.05), SBNP disabled for discrete target modes; FP32 training, Conservative global DualNorm rates: lr=0.003, D multiplier=1, prior multiplier=2; annealing starts at 0.25
- **Parent**: `stream-bispf-optimized`
- **Status**: `FAIL` (`0/4` required passes, qualified tier 0)
  - Tier 1 `seq_img_round_robin_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.15`; `hq >= 0.9`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, mean_rmse=0.3003, modes=0.0000
  - Tier 1 `seq_img_dwell_bars4` (required): **FAIL** — terminal 0/5 · failed `finite_template_tv <= 0.2`; `hq >= 0.85`; `modes == 4.0` · finite_template_tv=1.0000, hq=0.0000, mean_rmse=0.3006, modes=0.0000
  - Tier 1 `seq_img_translating_bar_next` (required): **FAIL** — terminal 0/5 · failed `direction_acc >= 0.9`; `next_frame_rmse <= 0.12` · direction_acc=0.5156, next_frame_rmse=0.3603
  - Tier 1 `seq_img_fork_rollout` (required): **FAIL** — terminal 0/5 · failed `branch_tv <= 0.2`; `hq >= 0.85`; `within_state_modes == 2.0` · branch_tv=1.0000, hq=0.0000, mean_rmse=0.3293, within_state_modes=0.0000


Interpretation and historical evidence: [research notes](research-notes.md).
