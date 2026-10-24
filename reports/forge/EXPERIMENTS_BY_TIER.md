# Experiments by Tier

| View | Tier | Task | Role | Batch | Particles | Steps | Prior | Gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `single_stream_stability` | 1 | `seq_img_round_robin_bars4` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `hq>=0.9`, `modes==4.0`, `finite_template_tv<=0.15` |
| `single_stream_stability` | 1 | `seq_img_dwell_bars4` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `hq>=0.85`, `modes==4.0`, `finite_template_tv<=0.2` |
| `single_stream_stability` | 1 | `seq_img_translating_bar_next` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `next_frame_rmse<=0.12`, `direction_acc>=0.9` |
| `single_stream_stability` | 1 | `seq_img_fork_rollout` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `hq>=0.85`, `within_state_modes==2.0`, `branch_tv<=0.2` |
