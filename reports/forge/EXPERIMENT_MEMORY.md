# Experiment Memory

- **Primary View**: [`single_stream_stability`](technique-inventory.md) (revision 1)
- **Tasks by Tier**: [`EXPERIMENTS_BY_TIER.md`](EXPERIMENTS_BY_TIER.md)

## Recorded Candidates

### `fast-weight-bcap-ns` (`Global-Stream-BCAP`)
- **Changed factors**: Non-saturating GAN loss for batch_size=1 stability, Cosine learning rate annealing (lr=0.005, betas=(0.0, 0.999), lr_schedule=cosine, lr_anneal_start=0.5, lr_floor=0.05), 64-frame rehearsal reservoir spanning full dwell=16 4-mode cycle, Temporal velocity direction memory with static noise gating (||diff|| > 0.35), Joint non-linear conditional critic architecture, Dwell-aware rehearsal replay and task-isolated context conditioning
- **Status**: `PASS` (`4/4` required passes on `cuda`)

### `bcap` (`BCAP`)
- **Changed factors**: Stateless BCAP baseline on batch_size=1 sequential stream
- **Status**: `FAIL` (`1/4` required passes on `cuda`)

### `fast-weight-bcap` (`FastWeight-BCAP`)
- **Changed factors**: Detached FastWeightMemory writer/reader context + bounded 64-frame reservoir replay on batch_size=1 stream
- **Status**: `FAIL` (`0/4` required passes on `cuda`)

### `ka2` (`KA2`)
- **Changed factors**: KA2 critic EMA anchor on batch_size=1 sequential stream
- **Status**: `FAIL` (`0/4` required passes on `cuda`)

## Baseline Candidate Autopsy

- **`bcap` (`1/4` PASS)**: Stateless baseline fails catastrophically on correlated blocks (`seq_img_dwell_bars4`, 0/5 passes, finite_template_tv=1.0) because without temporal memory it rapidly forgets past modes and collapses to the current dwelling mode. It also fails `seq_img_translating_bar_next` (0/5 passes, direction_acc=0.50) due to ambiguity of single-frame observations without velocity context.
- **`fast-weight-bcap` (`0/4` PASS)**: Incorporating fast Hebbian weights enables memory context, but without learning rate annealing and with single-sample rehearsal, large gradient steps at `batch_size=1` destabilize the discriminator and latent prior, causing mode collapse and high Total Variation across all tasks.
- **`ka2` (`0/4` PASS)**: Critic EMA anchors fail under non-stationary single-stream drift, causing rapid prior collapse (0/5 passes across all 4 tasks).

## Winning Global Recipe Dynamics: `fast-weight-bcap-ns` (`4/4` PASS)

A single unified global recipe qualifies on all 4 Tier 1 tasks (`4/4` PASS, 5/5 sustained terminal passes on each task) with zero per-task tuning:
- **Cosine Learning Rate Annealing**: Smooth annealing (`lr_schedule='cosine'`, `lr_anneal_start=0.5`, `lr_floor=0.05`) cools plasticity in the second half of training, stabilizing discriminator and latent distributions while preventing late-stage mode drift.
- **Multi-Sample Reservoir Rehearsal**: 7-sample rehearsal (`rehearsal_batch=7`) from a bounded 64-frame memory buffer prevents catastrophic forgetting across dwell blocks (`seq_img_dwell_bars4` hq=0.9688, tv=0.0938) while keeping the input stream strictly `batch_size=1`.
- **Temporal Velocity Context**: Difference-based velocity conditioning (`diff = frame - prev_frame`) with static noise gating (`||diff|| > 0.35`) disambiguates interior frames for `seq_img_translating_bar_next` (direction_acc=1.0000, rmse=0.0803).
- **Non-Saturating GAN Formulation**: Non-saturating loss stabilizes gradient flow and prevents discriminator saturation at batch_size=1.

## Recommendations for Tier 2

1. **Higher-Resolution Spatial Streams**: Scale memory-conditioned generators from 8x8 patches to 16x16 and 32x32 video sequences.
2. **Adaptive Rehearsal Allocation**: Investigate prioritized or surprise-driven reservoir sampling to maintain long-tail modes over 1,000+ step horizons.
3. **Continuous World-Model Latent Transport**: Couple fast-weight memory with continuous latent trajectory transport for complex multi-agent and physical environments.
