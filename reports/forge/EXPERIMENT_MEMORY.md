# Experiment Memory

- **Primary View**: [`single_stream_stability`](technique-inventory.md) (revision 1)
- **Tasks by Tier**: [`EXPERIMENTS_BY_TIER.md`](EXPERIMENTS_BY_TIER.md)

## Recorded Candidates

### `fast-weight-bcap-ns` (`Global-Stream-BCAP`)
- **Changed factors**: Non-saturating GAN loss for batch_size=1 stability, Constant Adam learning rate (lr=0.005, betas=(0.0, 0.999), lr_floor=1.0), 64-frame rehearsal reservoir spanning full dwell=16 4-mode cycle, Temporal velocity direction memory with static noise gating (||diff|| > 0.35), Joint non-linear conditional critic architecture, Dwell-aware rehearsal replay and task-isolated context conditioning
- **Status**: `FAIL` (`2/4` required passes on `cuda`)

### `bcap` (`BCAP`)
- **Changed factors**: Stateless BCAP baseline on batch_size=1 sequential stream
- **Status**: `FAIL` (`1/4` required passes on `cuda`)

### `fast-weight-bcap` (`FastWeight-BCAP`)
- **Changed factors**: Detached FastWeightMemory writer/reader context + bounded 64-frame reservoir replay on batch_size=1 stream
- **Status**: `FAIL` (`0/4` required passes on `cuda`)

### `ka2` (`KA2`)
- **Changed factors**: KA2 critic EMA anchor on batch_size=1 sequential stream
- **Status**: `FAIL` (`0/4` required passes on `cuda`)
