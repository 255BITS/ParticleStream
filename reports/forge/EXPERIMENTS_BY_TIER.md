# Experiments by Tier

| View | Tier | Task | Role | Batch | Particles | Steps | Prior | Gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `single_stream_stability` | 1 | `seq_img_round_robin_bars4` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `hq>=0.9`, `modes==4.0`, `finite_template_tv<=0.15` |
| `single_stream_stability` | 1 | `seq_img_dwell_bars4` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `hq>=0.85`, `modes==4.0`, `finite_template_tv<=0.2` |
| `single_stream_stability` | 1 | `seq_img_translating_bar_next` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `next_frame_rmse<=0.12`, `direction_acc>=0.9` |
| `single_stream_stability` | 1 | `seq_img_fork_rollout` | `required` | 1 | 32 | 480 | `mog` (σ=0.025) | `hq>=0.85`, `within_state_modes==2.0`, `branch_tv<=0.2` |

## Single-Stream Evaluation Protocol & Contracts

- **Batch Size Law**: Strictly online single-stream execution with `batch_size=1` and detached inputs (`requires_grad=False`).
- **Initialization**: Deterministic orthogonal parameter initialization (`particlegan.init`) under protocol seed 0.
- **RNG Stream Isolation**: Isolated `NamedStreams` (`init`, `data`, `prior`, `noise`, `eval`) where evaluation callbacks never mutate non-eval streams (`unintended_rng_deviations == 0`).
- **Evaluation Cadence & Budget**: Full budget of 480 training steps per task, evaluated every 20 steps (24 observation windows).
- **Sustained Terminal Gate**: Final 5 consecutive observation checks (steps 400, 420, 440, 460, 480) must simultaneously pass all metric bounds (`terminal_passing == 5`).
