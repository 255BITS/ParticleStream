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
| `single_stream_stability` | 1 | `seq_img_circle_orbit` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `orbit_rmse<=0.12`, `xy_rmse<=0.15`, `step_accuracy>=0.9`, `orbit_completion>=0.9`, `waypoint_modes==8.0` |
| `single_stream_stability` | 1 | `seq_img_square_circle_cycle` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `cycle_rmse<=0.12`, `s2c_rmse<=0.12`, `c2s_rmse<=0.12`, `xy_rmse<=0.15`, `step_accuracy>=0.9`, `cycle_completion>=0.9`, `waypoint_modes==8.0` |
| `single_stream_stability` | 1 | `seq_minimal_dense4` | `stretch` | 1 | 32 | 96 | `mog` (σ=0.025) | `valid_fraction>=0.9`, `modes==4.0`, `mass_tv<=0.15` |
| `single_stream_stability` | 1 | `seq_minimal_tokens4` | `stretch` | 1 | 32 | 96 | `mog` (σ=0.025) | `valid_fraction==1.0`, `modes==4.0`, `mass_tv<=0.1` |
| `single_stream_stability` | 2 | `seq_minimal_kernel2x2` | `stretch` | 1 | 32 | 192 | `mog` (σ=0.025) | `seen_response_rmse<=0.03`, `seen_max_abs_error<=0.05`, `heldout_response_rmse<=0.03`, `heldout_max_abs_error<=0.05` |
| `single_stream_stability` | 2 | `seq_minimal_addition2bit` | `stretch` | 1 | 32 | 192 | `mog` (σ=0.025) | `seen_exact_accuracy==1.0`, `seen_min_bit_confidence>=0.9`, `heldout_exact_accuracy==1.0`, `heldout_min_bit_confidence>=0.9` |
| `single_stream_stability` | 3 | `seq_minimal_velocity_extrapolation` | `stretch` | 1 | 32 | 384 | `mog` (σ=0.025) | `interpolation_centroid_rmse<=0.01`, `interpolation_speed_error<=0.01`, `interpolation_raw_rmse<=0.12`, `interpolation_point_validity==1.0`, `interpolation_direction_accuracy==1.0`, `extrapolation_centroid_rmse<=0.01`, `extrapolation_speed_error<=0.01`, `extrapolation_raw_rmse<=0.12`, `extrapolation_point_validity==1.0`, `extrapolation_direction_accuracy==1.0` |

## Declared Evaluation Contracts

| Task | Initializer | Sampling law | Observation steps | Sustained window | Timeout (s) |
| --- | --- | --- | --- | --- | --- |
| `seq_img_circle_orbit` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_delayed_cue` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_dwell_bars4` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_fork_rollout` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_persistent_fork` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_rare_mode_bars4` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_round_robin_bars4` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_square_circle_cycle` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_translating_bar_next` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_img_variable_dwell_bars4` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (20..480) | last 5: [400, 420, 440, 460, 480] | 300 |
| `seq_minimal_addition2bit` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (8..192) | last 5: [160, 168, 176, 184, 192] | 300 |
| `seq_minimal_dense4` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (4..96) | last 5: [80, 84, 88, 92, 96] | 300 |
| `seq_minimal_kernel2x2` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (8..192) | last 5: [160, 168, 176, 184, 192] | 300 |
| `seq_minimal_tokens4` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (4..96) | last 5: [80, 84, 88, 92, 96] | 300 |
| `seq_minimal_velocity_extrapolation` | `deterministic_orthogonal` | `enumerated_prior_clean` | 24 (16..384) | last 5: [320, 336, 352, 368, 384] | 300 |

## Minimal probe sampling

ParticleGAN/Forge 0.9.7 writes the generic `enumerated_prior_clean` label in its evaluation schema. For these probes, `host_definition.sampling` and the matching receipt artifact specify the actual latent law. All generated outputs omit additional output noise.

| Task | Latent law | Rows per case |
| --- | --- | --- |
| `seq_minimal_addition2bit` | `four_clean_centers_v1` | 4 |
| `seq_minimal_dense4` | `stratified_rows_frozen_mog_eps_v1` | 32 |
| `seq_minimal_kernel2x2` | `four_clean_centers_v1` | 4 |
| `seq_minimal_tokens4` | `stratified_rows_frozen_mog_eps_v1` | 32 |
| `seq_minimal_velocity_extrapolation` | `four_clean_centers_v1` | 4 |
