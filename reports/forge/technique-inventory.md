# Technique Inventory — `single_stream_stability` (`single_stream_stability` v1)

| Technique | Recipe | Compute | Tier 1 | Total | Status |
| --- | --- | --- | --- | --- | --- |
| **Global-Stream-BCAP** | `fast-weight-bcap-ns` | `cuda` | `4/4` | **`4/4`** | `PASS` |
| **BCAP** | `bcap` | `cuda` | `1/4` | **`1/4`** | `FAIL` |
| **FastWeight-BCAP** | `fast-weight-bcap` | `cuda` | `0/4` | **`0/4`** | `FAIL` |
| **KA2** | `ka2` | `cuda` | `0/4` | **`0/4`** | `FAIL` |

## Per-Task Breakdown

### Tier 1: `seq_img_round_robin_bars4` (required)
- **Goal**: Acquire and hold all 4 8x8 bar modes from a deterministic round-robin stream (batch_size=1, dwell=1).
- **Gates**: `hq >= 0.9`, `modes == 4.0`, `finite_template_tv <= 0.15`
  - `fast-weight-bcap-ns`: **PASS** (finite_template_tv=0.0312, hq=0.9688, mean_rmse=0.0356, modes=4.0000) · [goal.gif](gifs/fast-weight-bcap-ns_seq_img_round_robin_bars4.gif)
  - `bcap`: **FAIL** (finite_template_tv=0.2812, hq=0.7188, mean_rmse=0.1074, modes=4.0000) · [goal.gif](gifs/bcap_seq_img_round_robin_bars4.gif)
  - `fast-weight-bcap`: **FAIL** (finite_template_tv=0.2812, hq=0.7188, mean_rmse=0.1074, modes=4.0000) · [goal.gif](gifs/fast-weight-bcap_seq_img_round_robin_bars4.gif)
  - `ka2`: **FAIL** (finite_template_tv=0.3125, hq=0.6875, mean_rmse=0.1221, modes=4.0000) · [goal.gif](gifs/ka2_seq_img_round_robin_bars4.gif)

### Tier 1: `seq_img_dwell_bars4` (required)
- **Goal**: Retain all 4 8x8 bar modes across correlated blocks of dwell=16 identical frames (batch_size=1).
- **Gates**: `hq >= 0.85`, `modes == 4.0`, `finite_template_tv <= 0.2`
  - `fast-weight-bcap-ns`: **PASS** (finite_template_tv=0.0938, hq=0.9688, mean_rmse=0.0426, modes=4.0000) · [goal.gif](gifs/fast-weight-bcap-ns_seq_img_dwell_bars4.gif)
  - `bcap`: **FAIL** (finite_template_tv=1.0000, hq=0.0000, mean_rmse=0.4554, modes=0.0000) · [goal.gif](gifs/bcap_seq_img_dwell_bars4.gif)
  - `fast-weight-bcap`: **FAIL** (finite_template_tv=0.4375, hq=0.5625, mean_rmse=0.1407, modes=4.0000) · [goal.gif](gifs/fast-weight-bcap_seq_img_dwell_bars4.gif)
  - `ka2`: **FAIL** (finite_template_tv=0.8438, hq=0.1562, mean_rmse=0.2936, modes=2.0000) · [goal.gif](gifs/ka2_seq_img_dwell_bars4.gif)

### Tier 1: `seq_img_translating_bar_next` (required)
- **Goal**: Predict the next 8x8 frame of a bouncing bar from ambiguous interior frames using memory (batch_size=1).
- **Gates**: `next_frame_rmse <= 0.12`, `direction_acc >= 0.9`
  - `fast-weight-bcap-ns`: **PASS** (direction_acc=1.0000, next_frame_rmse=0.0803) · [goal.gif](gifs/fast-weight-bcap-ns_seq_img_translating_bar_next.gif)
  - `bcap`: **FAIL** (direction_acc=0.5000, next_frame_rmse=0.4518) · [goal.gif](gifs/bcap_seq_img_translating_bar_next.gif)
  - `fast-weight-bcap`: **FAIL** (direction_acc=0.9844, next_frame_rmse=0.0912) · [goal.gif](gifs/fast-weight-bcap_seq_img_translating_bar_next.gif)
  - `ka2`: **FAIL** (direction_acc=0.5000, next_frame_rmse=0.3818) · [goal.gif](gifs/ka2_seq_img_translating_bar_next.gif)

### Tier 1: `seq_img_fork_rollout` (required)
- **Goal**: Cover both Left and Right future branches from the same stem memory state across particles (batch_size=1).
- **Gates**: `hq >= 0.85`, `within_state_modes == 2.0`, `branch_tv <= 0.2`
  - `fast-weight-bcap-ns`: **PASS** (branch_tv=0.0938, hq=0.9375, mean_rmse=0.0612, within_state_modes=2.0000) · [goal.gif](gifs/fast-weight-bcap-ns_seq_img_fork_rollout.gif)
  - `bcap`: **PASS** (branch_tv=0.0625, hq=0.9375, mean_rmse=0.0666, within_state_modes=2.0000) · [goal.gif](gifs/bcap_seq_img_fork_rollout.gif)
  - `fast-weight-bcap`: **FAIL** (branch_tv=0.1250, hq=0.8750, mean_rmse=0.0716, within_state_modes=2.0000) · [goal.gif](gifs/fast-weight-bcap_seq_img_fork_rollout.gif)
  - `ka2`: **FAIL** (branch_tv=0.2188, hq=0.9062, mean_rmse=0.0596, within_state_modes=2.0000) · [goal.gif](gifs/ka2_seq_img_fork_rollout.gif)
