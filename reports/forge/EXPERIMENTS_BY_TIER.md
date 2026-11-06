# Experiments by Tier

| View | Tier | Task | Role | Batch | Particles | Steps | Prior | Gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `single_stream_stability` | 1 | `seq_img_round_robin_bars4` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `hq>=0.9`, `modes==4.0`, `finite_template_tv<=0.15` |
| `single_stream_stability` | 1 | `seq_img_dwell_bars4` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `hq>=0.85`, `modes==4.0`, `finite_template_tv<=0.2` |
| `single_stream_stability` | 1 | `seq_img_translating_bar_next` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `next_frame_rmse<=0.12`, `direction_acc>=0.9` |
| `single_stream_stability` | 1 | `seq_img_fork_rollout` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `hq>=0.85`, `within_state_modes==2.0`, `branch_tv<=0.2` |
| `single_stream_stability` | 1 | `seq_img_variable_dwell_bars4` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `hq>=0.85`, `modes==4.0`, `finite_template_tv<=0.2` |
| `single_stream_stability` | 1 | `seq_img_rare_mode_bars4` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `hq>=0.85`, `finite_template_tv<=0.15`, `support_modes==4.0`, `rare_mass>=0.03125`, `rare_mass<=0.09375` |
| `single_stream_stability` | 1 | `seq_img_delayed_cue` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `outcome_accuracy>=0.9`, `prediction_rmse<=0.12` |
| `single_stream_stability` | 1 | `seq_img_persistent_fork` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `hq>=0.85`, `within_state_modes==2.0`, `branch_tv<=0.2`, `trajectory_consistency>=0.95` |

## Declared Evaluation Contracts

| Task | Initializer | Sampling law | Observation steps | Sustained window | Timeout (s) |
| --- | --- | --- | --- | --- | --- |
| `seq_img_delayed_cue` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_dwell_bars4` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_fork_rollout` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_persistent_fork` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_rare_mode_bars4` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_round_robin_bars4` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_translating_bar_next` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_variable_dwell_bars4` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
