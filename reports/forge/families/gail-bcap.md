<!-- Generated Forge family report -->

# GAIL-BCAP

[← Technique leaderboard](../technique-inventory.md)

<a id="technique-overview"></a>
## Technique overview

GAIL-BCAP translates stabilized generative adversarial imitation control into ParticleStream's strict `batch_size=1` online world-modeling contract. It pairs a Hinge margin critic and fixed BCAP input-gradient cap with a prioritized survival-plus-style generator objective: unbounded linear Hinge generator slopes are canceled in favor of pixelwise BCE reconstruction minus a bounded sigmoid critic style bonus modulated by causal return advantage and active-prototype policy entropy. Frontier recipes upgrade conditional transport with precision-whitened two-pass Gram-Schmidt innovation subspaces, pre-birth debiased `float64` Sherman-Morrison RLS, Mahalanobis spike-and-slab prototype gating with consistent discrete-fork branch anchoring, unbucketed 512-point Matérn-3/2 kernel ridge regression (`KRR`), orthogonal previous-frame value readout and causal $\alpha$-$\beta$ barycentric kinematic tracking in `CausalStateMemory`, causal Empirical-Bayes Dirichlet-precision gated Fisher-Sobolev $H^1$ and quartic boundary calibration, 2D lattice-harmonic latent lifting with $\mathbb Z^2$ shift-equivariant moment regularization on localized wavepackets, and polynomial Primal-Dual Schedule-Free generator iterate averaging.

[Implementation](../../../particlestream/gail_prior.py) · [Mathematical formulation](gail-bcap.md#mathematical-formulation) · [Simplified pseudocode](gail-bcap.md#simplified-pseudocode) · [Training details](gail-bcap.md#training-details) · [Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)

<a id="mathematical-formulation"></a>
## Mathematical formulation

$G$ is the task-owned residual generator, $D$ the raw-score convolutional critic, and $P$ the frozen $K$-row `MoGParticlePrior` ($K=32$, $d_z=8$, intrinsic scale $\sigma=0.025$). At chronological arrival $t$, $x_t\in[0,1]^{1\times 8\times 8}$ is the single target frame, $c_{t-1}\in\mathbb R^{d_c}$ ($d_c=16$) is the detached causal-state context (`write_lr` $\eta_w\in\{0.5,0.35\}$, contrast gate $\tau_{\mathrm{con}}\in\{0.03,0.005,0.002\}$, optional previous-frame value weight $w_{\mathrm{pv}}\in\{0,1\}$, optional $\alpha$-$\beta$ barycentric kinematic gains $(\alpha_{\mathrm{kf}},\beta_{\mathrm{kf}})=(0.20,0.04)$), and $\phi_t(x_t)\in\mathbb R^{d_z}$ encodes $v_t=\bar x_t-c_0\mathbf 1_{64}$ ($c_0=0.5$) via fixed orthogonal rows ($K_{\mathrm{sub}}=0$, $s_f\in\{2,4\}$), $K_{\mathrm{sub}}=6$ precision-whitened Gram-Schmidt innovation rows ($\nu_0=0.10$, $s_f=2.1$) plus 2 deterministic JL rows, or 2D lattice-harmonic barycentric coordinates ($s_{\mathrm{wp}}=2.5$) on localized mass-conserving wavepackets ($\alpha_{\mathrm{wp},t}=1$). $\lambda_{\mathrm{rec}}\in\{10,100\}$ is the survival BCE weight, $w_{\mathrm{style}}\in\{0.1,0.05,0.002\}$ the bounded style bonus weight, $\sigma_{\mathrm{sig}}(u)=(1+e^{-u})^{-1}$ the logistic sigmoid, $\gamma_R=0.99$ the causal return discount, $R_t$ and $(b_t,v_t)$ the post-commit return trace and moments, $\alpha_A\in\{0.25,0\}$ the advantage scale, $\lambda_{\mathrm{ctx}}\in\{0,5\}$ the Markov bottleneck weight, $\eta_H=0.01$ the active-prototype policy entropy bonus, $(M_t,P_{t,r},\Sigma_t)$ the per-coordinate `float64` RLS and residual covariance state, $(K_{\mathrm{mat}},h_{\mathrm{res}},\lambda_{\mathrm{krr}},C_{\mathrm{krr}})$ the Matérn-3/2 KRR kernel ($h_{\mathrm{res}}\in\{1.15,0.72\}$, $\lambda_{\mathrm{krr}}\in\{0.06,0.025\}$, $C_{\mathrm{krr}}\in\{0,512\}$), $\alpha_{\mathrm{dir},t}\in(0,1]$ the causal Dirichlet-precision constant-field gate ($\sigma_\nabla=10^{-7}$, $w_{\mathrm{sob}}=36$, $\gamma_{\mathrm{bnd}}=0.85$, $\lambda_{\mathrm{dir,ctx}}=55$, $\Delta c_{\mathrm{dir}}=0.02$), $(w_{\mathrm{eq}},w_{\mathrm{mom}},w_{\mathrm{mse}},\lambda_{\mathrm{wp,ctx}},N_{\mathrm{sh}},\Delta c_{\mathrm{wp}})=(3.0,0.07,2.5,80,6,0.205)$ the wavepacket shift-equivariant calibration parameters, and $(z_t^{\mathrm{sf}},x_t^{\mathrm{sf}},y_t^{\mathrm{sf}})$ the Schedule-Free dual, primal, and interpolation generator states.

**Hinge-BCAP Critic, Prioritized Survival + Bounded Style Bonus, and Markov Bottleneck**

$$
\begin{aligned}\mathcal L_D&=\big(1-D(x_t,c_{t-1})\big)_++\big(1+D(\hat x_D,c_{t-1})\big)_++\frac{\lambda}{2}\Big[(\lVert g_r\rVert_2-\kappa)_+^2+(\lVert g_f\rVert_2-\kappa)_+^2\Big],\\g_{t-1}&=1+\alpha_A\tanh\!\left(\frac{R_{t-1}-b_{t-1}}{\sqrt{v_{t-1}+10^{-10}}}\right),\\\mathcal L_G&=\lambda_{\mathrm{rec}}\Big[\operatorname{BCE}(\hat x_G,x_t)-w_{\mathrm{style}}\,g_{t-1}\,\sigma_{\mathrm{sig}}\big(D(\hat x_G,c_{t-1})\big)\Big]+\lambda_{\mathrm{ctx}}\lVert W_{fc,\mathrm{ctx}}\rVert_F^2.\end{aligned}
$$

Canceling the default unbounded Hinge generator term $-D(\hat x_G,c_{t-1})$ prevents single-sample critic spikes from destabilizing the shared decoder; scaling $\lambda_{\mathrm{rec}}=100$ with $w_{\mathrm{style}}=0.002$ (`online-gail-bcap-subspace` onward) strengthens posterior pixel anchoring $10\times$ while keeping effective style weight $\lambda_{\mathrm{rec}}w_{\mathrm{style}}=0.20$ calibrated.

**Entropy-Regularized Active-Prototype Quantile Policy**

$$
\begin{aligned}\mathcal A_{b^*}&=\big\{m\in\{0,\dots,M_{\mathrm{occ}}-1\}:N_{b^*,m}>0\big\},\\\tilde p_{b^*,m}&=(1-\eta_H)\frac{N_{b^*,m}}{\sum_{j\in\mathcal A_{b^*}}N_{b^*,j}}+\eta_H\frac{\mathbb I(m\in\mathcal A_{b^*})}{|\mathcal A_{b^*}|},\\m^*(i)&=\min\!\left(m_{\max}(\mathcal A_{b^*}),\;\inf\left\{m\ge 0:\sum_{j=0}^m \tilde p_{b^*,j}>\frac{i+0.5}{K}\right\}\right).\end{aligned}
$$

Mixing $\eta_H=0.01$ uniform probability strictly over active prototypes $\mathcal A_{b^*}$ at the nearest context bucket $b^*$ prevents finite-sample lock-out on rare discrete branches (`5:5:5:1`) and stabilizes recurrent state-action buckets in multi-episode robotics.

**Precision-Whitened Innovation Subspace, Spike-and-Slab RLS, and Matérn-3/2 KRR Transport**

$$
\begin{aligned}r_t&=\big(I-U_{k_{t-1}}U_{k_{t-1}}^\top\big)^2 v_t,\qquad g_{k_{t-1}}=\big(\lVert r_t\rVert_2^2+\nu_0^2\big)^{-1/2},\qquad \kappa_{3/2}(d)=\left(1+\frac{\sqrt{3}\,d}{h_{\mathrm{res}}}\right)\exp\!\left(-\frac{\sqrt{3}\,d}{h_{\mathrm{res}}}\right),\\\hat r_{\mathrm{krr}}(c)&=k_{3/2}(c)^\top\!\left(K_{3/2}+\operatorname{diag}\!\left(\frac{\lambda_{\mathrm{krr}}}{\min(V_b,4)}\right)\right)^{-1}R_{\mathrm{rls}},\qquad \hat z_{\mathrm{base}}(c)=\hat z_{\mathrm{ctx}}(c)+(1-g_{\mathrm{disc},t})\hat r_{\mathrm{krr}}(c),\\z_i^{\mathrm{cond}}(c)&=\hat z_{\mathrm{base}}(c)+\gamma_t(c)\Big(\hat\mu_{m^*(i)}-\hat z_{\mathrm{ctx}}(k_{b^*}^{\mathrm{ctx}})\Big)+\epsilon_i.\end{aligned}
$$

In `online-gail-bcap-subspace` and `online-gail-bcap-schedule-free`, $K_{\mathrm{sub}}=6$ anchored Gram-Schmidt directions capture exact active target subspaces while solving an exact $B_t\times B_t$ (`B_t <= 64`) `float64` Matérn-3/2 kernel ridge system over unquantized per-bucket RLS residuals $R_{\mathrm{rls},b}=\bar z_b-M_t\bar k_b^{\mathrm{ctx}}$ eliminates Nadaraya-Watson cluster-density bias near contact.

**Horizon-Invariant Causal Exponential Cooling and Primal-Dual Schedule-Free Generator**

$$
\begin{aligned}\eta(t)&=\eta_0\,r_{\mathrm{dec}}^{t/S_{\mathrm{dec}}},\qquad c_t=\min\!\left(1,\,\max\!\left(\frac{k_{\mathrm{sf}}}{t+k_{\mathrm{sf}}-1},\,c_{\min}\right)\right),\\y_t^{\mathrm{sf}}&=(1-\beta_{\mathrm{sf}})z_{t-1}^{\mathrm{sf}}+\beta_{\mathrm{sf}}x_{t-1}^{\mathrm{sf}},\qquad z_t^{\mathrm{sf}}=z_{t-1}^{\mathrm{sf}}+\big(y_t^{\mathrm{sf},+}-y_t^{\mathrm{sf}}\big),\qquad x_t^{\mathrm{sf}}=(1-c_t)x_{t-1}^{\mathrm{sf}}+c_t z_t^{\mathrm{sf}}.\end{aligned}
$$

Replacing horizon-coupled cosine annealing with causal step-count cooling (`lr=0.014`, $r_{\mathrm{dec}}=0.55$, $S_{\mathrm{dec}}=100$) and evaluating single-arrival gradients at $y_t^{\mathrm{sf}}$ (`betas=[0, 0.999]`, $\beta_{\mathrm{sf}}=0.88$, $k_{\mathrm{sf}}=4$, $c_{\min}=0.02$) maintains fast binary-mask acquisition across 96-step minimal streams (`dense4`) while serving smooth 50-step primal averages $x_t^{\mathrm{sf}}$ across 480-step image and robotics streams.

**Pre-Birth Debiased Unbucketed KRR and Causal Dirichlet-Precision Fisher-Sobolev Calibration**

$$
\begin{aligned}\alpha_{\mathrm{dir},t}&=\left(1+\frac{\max(E_\nabla(x_t),\,\bar E_{\nabla,t-1})}{\sigma_\nabla^2}\right)^{-1},\qquad w_{\mathrm{bnd}}(x_t)=1+\gamma_{\mathrm{bnd}}(2\bar x_t-1)^4,\\\mathcal S_{\mathrm{sob}}(\hat x_G,x_t)&=\frac12\sum_{d\in\{\mathrm{row},\mathrm{col}\}}\operatorname{mean}\!\left(\frac{0.25}{\overline{\hat x_G(1-\hat x_G)}_d+0.015}\big(\nabla_d\hat x_G-\nabla_d x_t\big)^2\right),\\\mathcal L_G^{\mathrm{dir}}&=\lambda_{\mathrm{rec}}\Big[\big(1-\alpha_{\mathrm{dir},t}+\alpha_{\mathrm{dir},t}w_{\mathrm{bnd}}(x_t)\big)\operatorname{BCE}(\hat x_G,x_t)+\alpha_{\mathrm{dir},t}w_{\mathrm{sob}}\,\mathcal S_{\mathrm{sob}}(\hat x_G,x_t)-\tilde w_{\mathrm{style},t}g_{t-1}\sigma_{\mathrm{sig}}(D)\Big]+(\lambda_{\mathrm{ctx}}+\alpha_{\mathrm{dir},t}\lambda_{\mathrm{dir,ctx}})\lVert W_{fc,\mathrm{ctx}}\rVert_F^2.\end{aligned}
$$

In `online-gail-bcap-debiased-krr`, per-coordinate RLS covariances $P_{t,r}$ reset cleanly at subspace row birth, effective style weight $\tilde w_{\mathrm{style},t}=w_{\mathrm{style}}\max(g_{\mathrm{disc},t},\alpha_{\mathrm{dir},t})$ eliminates single-sample critic drift on continuous unimodal physics while preserving full adversarial crispness on discrete/multimodal and constant-field streams, and KRR evaluates up to $C_{\mathrm{krr}}=512$ unbucketed post-birth continuous observations (`0` stored frames), advancing `seq_robotics_fetch_slide` to `5/5`. In `online-gail-bcap-dirichlet-sobolev`, $\alpha_{\mathrm{dir},t}=1$ on spatially constant fields (`seq_minimal_kernel2x2`, $E_\nabla(x_t)=0$) and $\alpha_{\mathrm{dir},t}<10^{-7}$ on all spatially structured streams, suppressing $2\times 2$ transposed-conv ripple and context-bypass overfitting without perturbing any image, token, or robotics task.

**Causal Alpha-Beta Barycentric Kinematics, 2D Lattice-Harmonic Lifting, and Z2 Shift-Equivariant Moment Prior**

$$
\begin{aligned}\hat p_k&=(\hat p_{k-1}+\hat v_{k-1})+\alpha_{\mathrm{kf}}\big(b_k-(\hat p_{k-1}+\hat v_{k-1})\big),\qquad \hat v_k=\hat v_{k-1}+\beta_{\mathrm{kf}}\big(b_k-(\hat p_{k-1}+\hat v_{k-1})\big),\\\psi(u_x,u_y)&=s_{\mathrm{wp}}s_f\big[u_x,\,u_y,\,0.5\sin 2\pi s_+,\,0.5\cos 2\pi s_-,\,0.4\sin 7\pi s_+,\,0.4\cos 7\pi s_-,\,0.3\sin 14\pi s_+,\,0.3\cos 14\pi s_-\big]^\top,\\\mathcal M(\hat x,x)&=(m(\hat x)-m(x))^2+4\lVert \mu(\hat x)-\mu(x)\rVert_2^2+25\lVert \sigma^2(\hat x)-\sigma^2(x)\rVert_2^2,\\\mathcal L_{\mathrm{wp}}&=\mathcal L_{\mathrm{mom,mse}}(\hat x_G,x_t)+\frac{w_{\mathrm{eq}}}{|\mathcal S_t|}\sum_{\delta\in\mathcal S_t}\Big[\operatorname{BCE}\big(G(\phi(\mathsf T_\delta x_t),0),\mathsf T_\delta x_t\big)+\mathcal L_{\mathrm{mom,mse}}\big(G(\phi(\mathsf T_\delta x_t),0),\mathsf T_\delta x_t\big)\Big].\end{aligned}
$$

In `online-gail-bcap-kinematic-equivariant`, localized mass-conserving wavepackets ($m\in[0.25,2.5]$, participation ratio $\mathrm{PR}\in[1,6]$, $x\ge 0$) initialize exact 2- and 3-point MVUE velocity $\hat v_2=\frac12(b_2-b_0)$ and track $k\ge 3$ with $(\alpha_{\mathrm{kf}},\beta_{\mathrm{kf}})=(0.20,0.04)$ over relative-masked barycenters ($x\ge 0.015\max x$), yielding identically zero innovation on constant-velocity training streams while damping closed-loop autoregressive rollout jitter $5\times$. With $s_\pm=u_x\pm u_y$, lattice-period harmonics $(7\pi,14\pi)$ let $G$ compensate its $2\times 2\to 4\times 4\to 8\times 8$ nearest-upsample block phase, while $N_{\mathrm{sh}}=6$ exact interior $\mathbb Z^2$ shifts $\mathsf T_\delta x_t$ decouple unseen spatial positions from velocity ($8/8$ passing terminal checks on `seq_minimal_velocity_extrapolation`, `0` activation on all other 19 tasks).

<a id="simplified-pseudocode"></a>
## Simplified pseudocode

```text
Initialize fallback orthogonal encoder E_0, frozen MoG prior P, empty subspace/prototype/context/KRR buffers, P_0 = (1 / assoc_ridge) I, and Schedule-Free state (z_0 = x_0 = theta_G).
For each chronological single-arrival step t (batch_size = 1, replay = none):
  Before revealing x_t, serve y_hat = G_{x_{t-1}}(z_cond(i_serve, c_{t-1}), c_{t-1}) under an isolated eval RNG.
  Reveal single target frame x_t; stage Gram-Schmidt innovation row (or wavepacket harmonic lift if alpha_wp = 1) and encode phi_t(x_t).
  If schedule_free_beta > 0, interpolate generator parameters to y_t = (1 - beta_sf) z_{t-1} + beta_sf x_{t-1}.
  Draw uniform row i_D; form z_D via active-entropy quantiles + debiased RLS / harmonic lift / spike-and-slab + (1 - g_disc) * Matern32_KRR(c_{t-1}); step D with Hinge + BCAP cap.
  With D fixed, draw uniform row i_G and form posterior code z_G = phi_t(x_t) + eps_{i_G}; evaluate survival BCE (+ Dirichlet Fisher-Sobolev or wavepacket Z2 shift-equivariant moment prior if gated) - bounded style bonus + Markov penalty; step G.
  If schedule_free_beta > 0, update dual z_t = z_{t-1} + (y_t^+ - y_t) and primal x_t = (1 - c_t) x_{t-1} + c_t z_t, leaving G at x_t for causal serving.
  After optimizer updates succeed, commit staged innovation row, prototypes, context buckets, unbucketed KRR entry, Dirichlet/wavepacket counters, return baseline, and float64 RLS + covariance.
  Advance detached causal state c_t = CausalStateMemory.write(x_t) (with prev_val readout and optional alpha-beta kinematic tracker) without storing or replaying frames.
```

<a id="training-details"></a>
## Training details

| Characteristic | Behavior |
| --- | --- |
| Adversarial loss | Hinge margin adversarial critic loss (`loss="hinge"`), paired on $G$ with canceled linear critic slope (`cancel_linear_gan=True`), prioritized pixelwise BCE survival anchor ($\lambda_{\mathrm{rec}}=10$ in Hinge..SNR; $100$ in Subspace..Kinematic-Equivariant), bounded sigmoid style bonus ($w_{\mathrm{style}}\in\{0.1,0.05,0.002\}$), Markov context-bypass penalty (`context_bypass_weight=5.0`), causal Dirichlet-precision Fisher-Sobolev $H^1$ + quartic calibration (`Dirichlet-Sobolev` onward), and $\mathbb Z^2$ shift-equivariant central-moment + $L_2$ wavepacket regularization in `online-gail-bcap-kinematic-equivariant`. |
| Optimizer | Native Adam via `bcap_adam` (`betas=(0, 0.999)` in Hinge/BCE/Spike-Slab/SNR/Schedule-Free; `(0.5, 0.999)` in Momentum; `(0.4, 0.999)` in Subspace; `(0, 0.95)` paired with polynomial Primal-Dual Schedule-Free $\beta_{\mathrm{sf}}=0.88$ in Debiased-KRR, Dirichlet-Sobolev, and Kinematic-Equivariant). Base MoG storage table is frozen (`prior_update="frozen"`). |
| Learning rates and annealing | Hinge..SNR use `lr=0.005`, `critic_lr_mult=1.0`, cosine-annealed from `0.5` to `0.05`; `online-gail-bcap-subspace` uses `lr=0.007`, `d_lr_mult=0.02`, cosine-annealed from `0.55` to `0.06`; `online-gail-bcap-schedule-free` uses causal `lr_schedule="exponential"` (`lr=0.014`, `lr_decay_rate=0.55`, `lr_decay_steps=100`); Debiased-KRR, Dirichlet-Sobolev, and Kinematic-Equivariant use `lr=0.016`, `lr_decay_rate=0.58`, `lr_decay_steps=100`, `d_lr_mult=0.02`. |
| Parameter-gradient clipping | None (`critic_grad_guard=False`). Unbounded generator critic gradients are eliminated structurally via the bounded sigmoid style bonus and BCAP input-gradient cap. |
| Critic penalties and anchors | Fixed one-sided squared $L_2$ BCAP cap on real and conditional-prior fake inputs ($\lambda=1.0$, $\kappa=1.0$) every critic step; no EMA critic anchor. |
| Damping and update guards | No A2 sparse-row damping (`latent_damping_max_rate=0.0`), zero generator-center probes (`probe_law="no_generator_probes_v1"`), regularized Mahalanobis covariance $\tilde\Sigma_t\succeq 10^{-4}I$, and exact ridge-stabilized Matérn-3/2 KRR ($\lambda_{\mathrm{krr}}\in\{0.06,0.025\}$). |
| Online memory, prior and routing | Detached `CausalStateMemory` ($d_c=16$, `write_lr=0.5` in Hinge..SNR and `0.35` in Subspace..Kinematic-Equivariant, `association_weight=0.0`, `normalize_key=False, key_gain=0.5` from Spike-Slab onward, `contrast_threshold=0.002` from Subspace onward, `prev_val_weight=1.0` from Debiased-KRR onward, `barycentric_kinematics=True` with `kinematic_alpha=0.20, kinematic_beta=0.04` in Kinematic-Equivariant) paired with `GAILConditionalReconstruction` (`router="gail_conditional"`, `policy_entropy_bonus=0.01`, `subspace_dims=6`, `residual_kernel="matern32_krr"`, `krr_capacity=512`, `replay_policy="none"`, `rehearsal_batch=0`). |
| Training and sampling noise | Zero scheduled critic-input or generator-output additive noise (`0.0`). Intrinsic task MoG perturbation $\sigma=0.025$ is preserved on every conditional and posterior latent draw; precision-whitened innovation rows (`subspace_nu0=0.10`, `feature_scale=2.1`) and wavepacket harmonic scaling (`wavepacket_snr_scale=2.5`) amplify continuous physical SNR without clipping discrete binary images. |
| Parameter averaging and serving | Hinge..Subspace serve live weights (`ema_decay=0.0`, `serve_average=False`); Schedule-Free, Debiased-KRR, Dirichlet-Sobolev, and Kinematic-Equivariant serve the polynomial Primal-Dual Schedule-Free iterate $x_t^{\mathrm{sf}}$ while interrogating gradients at $y_t^{\mathrm{sf}}=0.12 z_{t-1}^{\mathrm{sf}}+0.88 x_{t-1}^{\mathrm{sf}}$. |

<a id="configuration-differences"></a>
## Configuration differences

- GAIL-BCAP forms a ten-recipe single-factor ablation ladder over `online-feature-prior-bce`: `online-gail-bcap-hinge` (10/10 required, 1/10 stretch) -> `online-gail-bcap-bce` (10/10 required, 1/10 stretch) -> `online-gail-bcap-momentum` -> `online-gail-bcap-spike-slab` (10/10 required, 3/10 stretch) -> `online-gail-bcap-snr` (10/10 required, 3/10 stretch) -> `online-gail-bcap-subspace` (10/10 required, 4/10 stretch) -> `online-gail-bcap-schedule-free` (10/10 required, 5/10 stretch) -> `online-gail-bcap-debiased-krr` (10/10 required, 6/10 stretch) -> `online-gail-bcap-dirichlet-sobolev` (10/10 required, 7/10 stretch) -> `online-gail-bcap-kinematic-equivariant` (10/10 required, 8/10 stretch).
- `online-gail-bcap-subspace` upgrades `online-gail-bcap-snr` with precision-whitened anchored Gram-Schmidt innovation subspaces (`subspace_dims=6`), `float64` Matérn-3/2 KRR over per-bucket RLS residuals (`residual_kernel="matern32_krr"`), fine-salience action gating (`contrast_threshold=0.002`), and calmed Hinge-BCAP calibration (`d_lr_mult=0.02`, `reconstruction_weight=100.0`, `style_bonus_weight=0.002`), passing `seq_robotics_fetch_pick_and_place` (`5/5`).
- `online-gail-bcap-schedule-free` upgrades `online-gail-bcap-subspace` with causal exponential cooling (`lr_schedule="exponential"`, `lr=0.014`, `lr_decay_rate=0.55`) and polynomial Primal-Dual Schedule-Free averaging (`betas=[0.0, 0.999]`, `schedule_free_beta=0.88`, `schedule_free_poly_k=4.0`, `schedule_free_c_min=0.02`), reaching `15/20` (`+ seq_minimal_dense4` `5/5`).
- `online-gail-bcap-debiased-krr` upgrades `online-gail-bcap-schedule-free` with orthogonal previous-frame value readout (`prev_val_weight=1.0`), pre-birth subspace RLS/KRR zero-debiasing (`subspace_birth_debias=True`, `subspace_rel_tol=0.055`, `assoc_ridge=0.08`), unbucketed 512-point Matérn-3/2 KRR (`krr_capacity=512`, `residual_bandwidth=0.72`, `residual_ridge=0.025`), consistent discrete-fork control-variate branch anchoring (`consistent_fork_branch=True`), discreteness-gated style bonus (`disc_gate_style=True`), and short-horizon Adam $\beta_2=0.95$ Schedule-Free cooling (`lr=0.016`, `lr_decay_rate=0.58`, `schedule_free_poly_k=2.4`, `schedule_free_c_min=0.005`), reaching `16/20` (`+ seq_robotics_fetch_slide` `5/5`).
- `online-gail-bcap-dirichlet-sobolev` is a strict single-factor child of `online-gail-bcap-debiased-krr` adding causal Empirical-Bayes Dirichlet-precision gated Fisher-Sobolev $H^1$ regularization (`sobolev_weight=36.0`, `sobolev_sigma=1e-7`), quartic boundary logit calibration (`boundary_quartic_gain=0.85`), Dirichlet Markov bottleneck shrinkage (`dirichlet_bypass_weight=55.0`), and Schedule-Free polynomial/floor boost (`dirichlet_c_boost=0.02`), reaching `17/20` (`+ seq_minimal_kernel2x2` `5/5`).
- `online-gail-bcap-kinematic-equivariant` is a strict single-factor child of `online-gail-bcap-dirichlet-sobolev` adding causal $\alpha$-$\beta$ barycentric kinematic tracking (`barycentric_kinematics=True`, `kinematic_alpha=0.2`, `kinematic_beta=0.04`), 2D lattice-harmonic barycentric latent lifting (`wavepacket_snr_scale=2.5`), and $\mathbb Z^2$ shift-equivariant central-moment + $L_2$ decoder calibration (`wavepacket_equiv_weight=3.0`, `wavepacket_moment_weight=0.07`, `wavepacket_mse_weight=2.5`, `wavepacket_bypass_weight=80.0`, `wavepacket_shifts=6`, `wavepacket_c_boost=0.205`), ranking #1 overall on `single_stream_stability` view 10 at `18/20` (`10/10` required + `8/10` stretch: conquering Tier 3 `seq_minimal_velocity_extrapolation` `5/5` alongside all 4 Tier 2 minimal tasks and 3 Tier 3 Fetch robotics tasks).
- Executed strictly with `batch_size=1` (real and fake), `seed=0`, `replay_policy="none"`, `rehearsal_batch=0`, and zero generator probes across all 20 tasks.

<details>
<summary>Implementation and recipe sources</summary>

These links support the explanation. Recorded results below remain bound to their own executed source.

- [particlestream/gail_prior.py](../../../particlestream/gail_prior.py)
- [particlestream/transport.py](../../../particlestream/transport.py)
- [particlestream/feature_prior.py](../../../particlestream/feature_prior.py)
- [particlestream/memory.py](../../../particlestream/memory.py)
- [particlestream/tier1.py](../../../particlestream/tier1.py)
- [configs/forge/ideas/online-gail-bcap-hinge.json](../../../configs/forge/ideas/online-gail-bcap-hinge.json)
- [configs/forge/ideas/online-gail-bcap-bce.json](../../../configs/forge/ideas/online-gail-bcap-bce.json)
- [configs/forge/ideas/online-gail-bcap-momentum.json](../../../configs/forge/ideas/online-gail-bcap-momentum.json)
- [configs/forge/ideas/online-gail-bcap-spike-slab.json](../../../configs/forge/ideas/online-gail-bcap-spike-slab.json)
- [configs/forge/ideas/online-gail-bcap-snr.json](../../../configs/forge/ideas/online-gail-bcap-snr.json)
- [configs/forge/ideas/online-gail-bcap-subspace.json](../../../configs/forge/ideas/online-gail-bcap-subspace.json)
- [configs/forge/ideas/online-gail-bcap-schedule-free.json](../../../configs/forge/ideas/online-gail-bcap-schedule-free.json)
- [configs/forge/ideas/online-gail-bcap-debiased-krr.json](../../../configs/forge/ideas/online-gail-bcap-debiased-krr.json)
- [configs/forge/ideas/online-gail-bcap-dirichlet-sobolev.json](../../../configs/forge/ideas/online-gail-bcap-dirichlet-sobolev.json)
- [configs/forge/ideas/online-gail-bcap-kinematic-equivariant.json](../../../configs/forge/ideas/online-gail-bcap-kinematic-equivariant.json)

</details>

Each section below describes one exact global recipe. Configuration and parent deltas come from the declarations; results come from Forge's current source-bound receipts. Required and stretch evidence stay separate. A final metric can meet its gate while the sustained verdict fails: every one of the last five checks must pass.

The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording in declaration history describes provenance; the effective configuration below controls execution.

<a id="online-gail-bcap-kinematic-equivariant"></a>
## `online-gail-bcap-kinematic-equivariant`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-kinematic-equivariant.json)

Parent: [`online-gail-bcap-dirichlet-sobolev`](gail-bcap.md#online-gail-bcap-dirichlet-sobolev).

Declared changes:

- Causal alpha-beta barycentric kinematic tracker (barycentric_kinematics=true, kinematic_alpha=0.2, kinematic_beta=0.04), 2D lattice-harmonic barycentric latent lifting (wavepacket_snr_scale=2.5), and Z^2 shift-equivariant central-moment + L2 decoder regularization (wavepacket_equiv_weight=3.0, wavepacket_moment_weight=0.07, wavepacket_mse_weight=2.5, wavepacket_bypass_weight=80.0, wavepacket_shifts=6, wavepacket_c_boost=0.205) gated strictly on localized mass-conserving wavepackets

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.barycentric_kinematics` | _unset_ | `true` |
| `memory_policy.kinematic_alpha` | _unset_ | `0.2` |
| `memory_policy.kinematic_beta` | _unset_ | `0.04` |
| `memory_policy.transport.wavepacket_bypass_weight` | _unset_ | `80.0` |
| `memory_policy.transport.wavepacket_c_boost` | _unset_ | `0.205` |
| `memory_policy.transport.wavepacket_equiv_weight` | _unset_ | `3.0` |
| `memory_policy.transport.wavepacket_moment_weight` | _unset_ | `0.07` |
| `memory_policy.transport.wavepacket_mse_weight` | _unset_ | `2.5` |
| `memory_policy.transport.wavepacket_shifts` | _unset_ | `6` |
| `memory_policy.transport.wavepacket_snr_scale` | _unset_ | `2.5` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "barycentric_kinematics": true,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.002,
    "decay": 0.9,
    "dim": 16,
    "key_gain": 0.5,
    "kind": "fast_weight",
    "kinematic_alpha": 0.2,
    "kinematic_beta": 0.04,
    "normalize_key": false,
    "prev_val_weight": 1.0,
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "advantage_scale": 0.0,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 1.0,
      "assoc_ridge": 0.08,
      "boundary_quartic_gain": 0.85,
      "cancel_linear_gan": true,
      "consistent_fork_branch": true,
      "context_bypass_weight": 5.0,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "covariance_ema": 0.05,
      "covariance_eps": 0.0001,
      "dirichlet_bypass_weight": 55.0,
      "dirichlet_c_boost": 0.02,
      "disc_gate_style": true,
      "feature_scale": 2.1,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "kernel_bandwidth": 0.35,
      "krr_capacity": 512,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 100.0,
      "residual_bandwidth": 0.72,
      "residual_kernel": "matern32_krr",
      "residual_ridge": 0.025,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "schedule_free_beta": 0.88,
      "schedule_free_c_min": 0.005,
      "schedule_free_poly_k": 2.4,
      "sobolev_sigma": 1e-07,
      "sobolev_weight": 36.0,
      "spike_slab_scale": 1.0,
      "style_bonus_weight": 0.002,
      "subspace_abs_tol": 0.008,
      "subspace_birth_debias": true,
      "subspace_dims": 6,
      "subspace_nu0": 0.1,
      "subspace_rel_tol": 0.055,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0,
      "variance_ratio_scale": 0.15,
      "visit_shrinkage": 0.25,
      "wavepacket_bypass_weight": 80.0,
      "wavepacket_c_boost": 0.205,
      "wavepacket_equiv_weight": 3.0,
      "wavepacket_moment_weight": 0.07,
      "wavepacket_mse_weight": 2.5,
      "wavepacket_shifts": 6,
      "wavepacket_snr_scale": 2.5
    },
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.95
    ],
    "d_lr_mult": 0.02,
    "loss": "hinge",
    "lr": 0.016,
    "lr_decay_rate": 0.58,
    "lr_decay_steps": 100,
    "lr_schedule": "exponential",
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 8/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `250be395355f9ac93b461304d505dfbc9f598bbb58ae7a6014a8af15aa132a67`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-kinematic-equivariant-required"></a>
### Required evidence

<a id="online-gail-bcap-kinematic-equivariant-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_img_round_robin_bars4.gif) | 2026-11-14T03:27:12.867+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_img_dwell_bars4.gif) | 2026-11-14T03:27:22.227+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.00433`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_img_translating_bar_next.gif) | 2026-11-14T03:27:37.810+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_img_fork_rollout.gif) | 2026-11-14T03:28:04.707+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_img_variable_dwell_bars4.gif) | 2026-11-14T03:28:19.940+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_img_rare_mode_bars4.gif) | 2026-11-14T03:28:34.948+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.009172` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_img_delayed_cue.gif) | 2026-11-14T03:29:01.473+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.09375`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_img_persistent_fork.gif) | 2026-11-14T03:30:19.248+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.010845`; `xy_rmse=0.002588`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_img_circle_orbit.gif) | 2026-11-14T03:31:42.764+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.010161`; `s2c_rmse=0.010236`; `c2s_rmse=0.010086`; `xy_rmse=0.002449`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_img_square_circle_cycle.gif) | 2026-11-14T03:34:16.401+00:00 |

<a id="online-gail-bcap-kinematic-equivariant-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-kinematic-equivariant-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_minimal_dense4.gif) | 2026-11-14T03:34:20.616+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_minimal_tokens4.gif) | 2026-11-14T03:34:24.735+00:00 |

<a id="online-gail-bcap-kinematic-equivariant-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **PASS** | 5/5 | `seen_response_rmse=0.018992`; `seen_max_abs_error=0.040003`; `heldout_response_rmse=0.012883`; `heldout_max_abs_error=0.030912` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_minimal_kernel2x2.gif) | 2026-11-14T03:34:33.659+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **PASS** | 5/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.99973`; `heldout_exact_accuracy=1`; `heldout_min_bit_confidence=0.99973` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_minimal_addition2bit.gif) | 2026-11-14T03:34:42.405+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **PASS** | 5/5 | `position_delta_rmse_m=0.0015976`; `relative_mse=0.0090819`; `moving_direction_accuracy=0.95833`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_robotics_fetch_reach.gif) | 2026-11-14T03:35:05.565+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010996`; `relative_mse=0.80557`; `moving_direction_accuracy=0.67442`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_robotics_fetch_push.gif) | 2026-11-14T03:35:29.120+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **PASS** | 5/5 | `position_delta_rmse_m=0.0044738`; `relative_mse=0.12182`; `moving_direction_accuracy=0.88235`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_robotics_fetch_slide.gif) | 2026-11-14T03:35:52.922+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **PASS** | 5/5 | `position_delta_rmse_m=0.0014521`; `relative_mse=0.013434`; `moving_direction_accuracy=0.91228`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_robotics_fetch_pick_and_place.gif) | 2026-11-14T03:36:17.499+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0075249`; `relative_mse=0.83755`; `moving_direction_accuracy=0.65421`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_robotics_hand_reach.gif) | 2026-11-14T03:36:40.358+00:00 |

<a id="online-gail-bcap-kinematic-equivariant-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **PASS** | 5/5 | `interpolation_centroid_rmse=0.0039757`; `interpolation_speed_error=0.0080746`; `interpolation_raw_rmse=0.0034644`; `interpolation_point_validity=1`; `interpolation_direction_accuracy=1`; `extrapolation_centroid_rmse=0.005258`; `extrapolation_speed_error=0.0047053`; `extrapolation_raw_rmse=0.0042621`; `extrapolation_point_validity=1`; `extrapolation_direction_accuracy=1` | [receipt](../receipts/cuda/online-gail-bcap-kinematic-equivariant/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-kinematic-equivariant_seq_minimal_velocity_extrapolation.gif) | 2026-11-14T03:37:08.348+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-kinematic-equivariant --device cuda`.

<a id="online-gail-bcap-dirichlet-sobolev"></a>
## `online-gail-bcap-dirichlet-sobolev`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-dirichlet-sobolev.json)

Parent: [`online-gail-bcap-debiased-krr`](gail-bcap.md#online-gail-bcap-debiased-krr).

Declared changes:

- Causal Empirical-Bayes Dirichlet-precision gated Fisher-Sobolev H1 seminorm (sobolev_weight=36.0, sobolev_sigma=1e-7), quartic extremal boundary logit calibration (boundary_quartic_gain=0.85), Dirichlet Markov bypass shrinkage (dirichlet_bypass_weight=55.0), and Schedule-Free polynomial/floor boost (dirichlet_c_boost=0.02) suppressing 2x2 convolutional ripple and context-bypass memorization on constant-field responses with zero perturbation on spatially structured streams

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.transport.boundary_quartic_gain` | _unset_ | `0.85` |
| `memory_policy.transport.dirichlet_bypass_weight` | _unset_ | `55.0` |
| `memory_policy.transport.dirichlet_c_boost` | _unset_ | `0.02` |
| `memory_policy.transport.sobolev_sigma` | _unset_ | `1e-07` |
| `memory_policy.transport.sobolev_weight` | _unset_ | `36.0` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.002,
    "decay": 0.9,
    "dim": 16,
    "key_gain": 0.5,
    "kind": "fast_weight",
    "normalize_key": false,
    "prev_val_weight": 1.0,
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "advantage_scale": 0.0,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 1.0,
      "assoc_ridge": 0.08,
      "boundary_quartic_gain": 0.85,
      "cancel_linear_gan": true,
      "consistent_fork_branch": true,
      "context_bypass_weight": 5.0,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "covariance_ema": 0.05,
      "covariance_eps": 0.0001,
      "dirichlet_bypass_weight": 55.0,
      "dirichlet_c_boost": 0.02,
      "disc_gate_style": true,
      "feature_scale": 2.1,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "kernel_bandwidth": 0.35,
      "krr_capacity": 512,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 100.0,
      "residual_bandwidth": 0.72,
      "residual_kernel": "matern32_krr",
      "residual_ridge": 0.025,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "schedule_free_beta": 0.88,
      "schedule_free_c_min": 0.005,
      "schedule_free_poly_k": 2.4,
      "sobolev_sigma": 1e-07,
      "sobolev_weight": 36.0,
      "spike_slab_scale": 1.0,
      "style_bonus_weight": 0.002,
      "subspace_abs_tol": 0.008,
      "subspace_birth_debias": true,
      "subspace_dims": 6,
      "subspace_nu0": 0.1,
      "subspace_rel_tol": 0.055,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0,
      "variance_ratio_scale": 0.15,
      "visit_shrinkage": 0.25
    },
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.95
    ],
    "d_lr_mult": 0.02,
    "loss": "hinge",
    "lr": 0.016,
    "lr_decay_rate": 0.58,
    "lr_decay_steps": 100,
    "lr_schedule": "exponential",
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 7/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `04100a48f4c807277b1724817cae0485aef9a9daa4224b909c2a2a2cbdcf4815`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-dirichlet-sobolev-required"></a>
### Required evidence

<a id="online-gail-bcap-dirichlet-sobolev-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_img_round_robin_bars4.gif) | 2026-11-14T03:13:14.579+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_img_dwell_bars4.gif) | 2026-11-14T03:13:28.963+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.00433`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_img_translating_bar_next.gif) | 2026-11-14T03:13:51.257+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_img_fork_rollout.gif) | 2026-11-14T03:14:16.743+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_img_variable_dwell_bars4.gif) | 2026-11-14T03:14:31.759+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_img_rare_mode_bars4.gif) | 2026-11-14T03:14:47.020+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.009172` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_img_delayed_cue.gif) | 2026-11-14T03:15:08.476+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.09375`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_img_persistent_fork.gif) | 2026-11-14T03:16:27.037+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.010845`; `xy_rmse=0.002588`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_img_circle_orbit.gif) | 2026-11-14T03:17:51.167+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.010161`; `s2c_rmse=0.010236`; `c2s_rmse=0.010086`; `xy_rmse=0.002449`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_img_square_circle_cycle.gif) | 2026-11-14T03:20:30.081+00:00 |

<a id="online-gail-bcap-dirichlet-sobolev-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-dirichlet-sobolev-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_minimal_dense4.gif) | 2026-11-14T03:20:33.928+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_minimal_tokens4.gif) | 2026-11-14T03:20:37.951+00:00 |

<a id="online-gail-bcap-dirichlet-sobolev-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **PASS** | 5/5 | `seen_response_rmse=0.018992`; `seen_max_abs_error=0.040003`; `heldout_response_rmse=0.012883`; `heldout_max_abs_error=0.030912` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_minimal_kernel2x2.gif) | 2026-11-14T03:20:46.464+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **PASS** | 5/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.99973`; `heldout_exact_accuracy=1`; `heldout_min_bit_confidence=0.99973` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_minimal_addition2bit.gif) | 2026-11-14T03:20:54.902+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **PASS** | 5/5 | `position_delta_rmse_m=0.0015976`; `relative_mse=0.0090819`; `moving_direction_accuracy=0.95833`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_robotics_fetch_reach.gif) | 2026-11-14T03:21:16.940+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010996`; `relative_mse=0.80557`; `moving_direction_accuracy=0.67442`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_robotics_fetch_push.gif) | 2026-11-14T03:21:39.587+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **PASS** | 5/5 | `position_delta_rmse_m=0.0044738`; `relative_mse=0.12182`; `moving_direction_accuracy=0.88235`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_robotics_fetch_slide.gif) | 2026-11-14T03:22:02.556+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **PASS** | 5/5 | `position_delta_rmse_m=0.0014521`; `relative_mse=0.013434`; `moving_direction_accuracy=0.91228`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_robotics_fetch_pick_and_place.gif) | 2026-11-14T03:22:25.947+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0075249`; `relative_mse=0.83755`; `moving_direction_accuracy=0.65421`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_robotics_hand_reach.gif) | 2026-11-14T03:22:48.011+00:00 |

<a id="online-gail-bcap-dirichlet-sobolev-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.10058`; `interpolation_speed_error=0.07278`; `interpolation_raw_rmse=0.064798`; `interpolation_point_validity=0.45833`; `interpolation_direction_accuracy=0.83333`; `extrapolation_centroid_rmse=0.12037`; `extrapolation_speed_error=0.15241`; `extrapolation_raw_rmse=0.056619`; `extrapolation_point_validity=0.60417`; `extrapolation_direction_accuracy=0.95833`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-dirichlet-sobolev/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-dirichlet-sobolev_seq_minimal_velocity_extrapolation.gif) | 2026-11-14T03:23:09.844+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-dirichlet-sobolev --device cuda`.

<a id="online-gail-bcap-debiased-krr"></a>
## `online-gail-bcap-debiased-krr`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-debiased-krr.json)

Parent: [`online-gail-bcap-schedule-free`](gail-bcap.md#online-gail-bcap-schedule-free).

Declared changes:

- Orthogonal previous-frame value readout (prev_val_weight=1.0), pre-birth subspace RLS/KRR zero-debiasing (subspace_birth_debias=true, subspace_rel_tol=0.055, assoc_ridge=0.08), unbucketed 512-point Matern-3/2 KRR dictionary (krr_capacity=512, residual_bandwidth=0.72, residual_ridge=0.025), consistent discrete-fork control-variate branch anchoring (consistent_fork_branch=true), discreteness-gated style bonus (disc_gate_style=true), and short-horizon Adam beta2 cooling (betas=[0.0, 0.95], lr=0.016, lr_decay_rate=0.58, schedule_free_poly_k=2.4, schedule_free_c_min=0.005)

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.prev_val_weight` | _unset_ | `1.0` |
| `memory_policy.transport.assoc_ridge` | `0.1` | `0.08` |
| `memory_policy.transport.consistent_fork_branch` | _unset_ | `true` |
| `memory_policy.transport.disc_gate_style` | _unset_ | `true` |
| `memory_policy.transport.krr_capacity` | _unset_ | `512` |
| `memory_policy.transport.residual_bandwidth` | `1.15` | `0.72` |
| `memory_policy.transport.residual_ridge` | `0.06` | `0.025` |
| `memory_policy.transport.schedule_free_c_min` | `0.02` | `0.005` |
| `memory_policy.transport.schedule_free_poly_k` | `4.0` | `2.4` |
| `memory_policy.transport.subspace_birth_debias` | _unset_ | `true` |
| `memory_policy.transport.subspace_rel_tol` | `0.12` | `0.055` |
| `recipe_overrides.betas` | `[0.0, 0.999]` | `[0.0, 0.95]` |
| `recipe_overrides.lr` | `0.014` | `0.016` |
| `recipe_overrides.lr_decay_rate` | `0.55` | `0.58` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.002,
    "decay": 0.9,
    "dim": 16,
    "key_gain": 0.5,
    "kind": "fast_weight",
    "normalize_key": false,
    "prev_val_weight": 1.0,
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "advantage_scale": 0.0,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 1.0,
      "assoc_ridge": 0.08,
      "cancel_linear_gan": true,
      "consistent_fork_branch": true,
      "context_bypass_weight": 5.0,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "covariance_ema": 0.05,
      "covariance_eps": 0.0001,
      "disc_gate_style": true,
      "feature_scale": 2.1,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "kernel_bandwidth": 0.35,
      "krr_capacity": 512,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 100.0,
      "residual_bandwidth": 0.72,
      "residual_kernel": "matern32_krr",
      "residual_ridge": 0.025,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "schedule_free_beta": 0.88,
      "schedule_free_c_min": 0.005,
      "schedule_free_poly_k": 2.4,
      "spike_slab_scale": 1.0,
      "style_bonus_weight": 0.002,
      "subspace_abs_tol": 0.008,
      "subspace_birth_debias": true,
      "subspace_dims": 6,
      "subspace_nu0": 0.1,
      "subspace_rel_tol": 0.055,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0,
      "variance_ratio_scale": 0.15,
      "visit_shrinkage": 0.25
    },
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.95
    ],
    "d_lr_mult": 0.02,
    "loss": "hinge",
    "lr": 0.016,
    "lr_decay_rate": 0.58,
    "lr_decay_steps": 100,
    "lr_schedule": "exponential",
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 6/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `a14066d191780029061ca8c25c65cb5a4920ccd917b59a85f8c2df74d140b842`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-debiased-krr-required"></a>
### Required evidence

<a id="online-gail-bcap-debiased-krr-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_img_round_robin_bars4.gif) | 2026-11-14T03:06:20.060+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_img_dwell_bars4.gif) | 2026-11-14T03:06:28.676+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.00433`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_img_translating_bar_next.gif) | 2026-11-14T03:06:43.037+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_img_fork_rollout.gif) | 2026-11-14T03:06:57.232+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_img_variable_dwell_bars4.gif) | 2026-11-14T03:07:05.964+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_img_rare_mode_bars4.gif) | 2026-11-14T03:07:14.556+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.008586` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_img_delayed_cue.gif) | 2026-11-14T03:07:29.002+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.09375`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_img_persistent_fork.gif) | 2026-11-14T03:08:09.065+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.010845`; `xy_rmse=0.002588`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_img_circle_orbit.gif) | 2026-11-14T03:08:56.057+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.010161`; `s2c_rmse=0.010236`; `c2s_rmse=0.010086`; `xy_rmse=0.002449`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_img_square_circle_cycle.gif) | 2026-11-14T03:10:26.678+00:00 |

<a id="online-gail-bcap-debiased-krr-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-debiased-krr-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_minimal_dense4.gif) | 2026-11-14T03:10:30.770+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_minimal_tokens4.gif) | 2026-11-14T03:10:34.571+00:00 |

<a id="online-gail-bcap-debiased-krr-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.02611`; `seen_max_abs_error=0.097214`; `heldout_response_rmse=0.026523`; `heldout_max_abs_error=0.06841`; failed terminal bounds: heldout_max_abs_error <= 0.05; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_minimal_kernel2x2.gif) | 2026-11-14T03:10:42.915+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **PASS** | 5/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.99993`; `heldout_exact_accuracy=1`; `heldout_min_bit_confidence=0.99995` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_minimal_addition2bit.gif) | 2026-11-14T03:10:51.155+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **PASS** | 5/5 | `position_delta_rmse_m=0.0015976`; `relative_mse=0.0090819`; `moving_direction_accuracy=0.95833`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_robotics_fetch_reach.gif) | 2026-11-14T03:11:12.902+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010996`; `relative_mse=0.80557`; `moving_direction_accuracy=0.67442`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_robotics_fetch_push.gif) | 2026-11-14T03:11:35.216+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **PASS** | 5/5 | `position_delta_rmse_m=0.0044738`; `relative_mse=0.12182`; `moving_direction_accuracy=0.88235`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_robotics_fetch_slide.gif) | 2026-11-14T03:11:57.699+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **PASS** | 5/5 | `position_delta_rmse_m=0.0014521`; `relative_mse=0.013434`; `moving_direction_accuracy=0.91228`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_robotics_fetch_pick_and_place.gif) | 2026-11-14T03:12:21.057+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0075249`; `relative_mse=0.83755`; `moving_direction_accuracy=0.65421`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_robotics_hand_reach.gif) | 2026-11-14T03:12:42.697+00:00 |

<a id="online-gail-bcap-debiased-krr-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.10058`; `interpolation_speed_error=0.07278`; `interpolation_raw_rmse=0.064798`; `interpolation_point_validity=0.45833`; `interpolation_direction_accuracy=0.83333`; `extrapolation_centroid_rmse=0.12037`; `extrapolation_speed_error=0.15241`; `extrapolation_raw_rmse=0.056619`; `extrapolation_point_validity=0.60417`; `extrapolation_direction_accuracy=0.95833`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-debiased-krr/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-debiased-krr_seq_minimal_velocity_extrapolation.gif) | 2026-11-14T03:13:04.371+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-debiased-krr --device cuda`.

<a id="online-gail-bcap-schedule-free"></a>
## `online-gail-bcap-schedule-free`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-schedule-free.json)

Parent: [`online-gail-bcap-subspace`](gail-bcap.md#online-gail-bcap-subspace).

Declared changes:

- Horizon-invariant causal exponential step-size cooling (lr_schedule=exponential, lr=0.014, lr_decay_rate=0.55, lr_decay_steps=100, betas=[0.0, 0.999]) paired with polynomial primal-dual Schedule-Free generator optimization (schedule_free_beta=0.88, schedule_free_poly_k=4.0, schedule_free_c_min=0.02)

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.transport.schedule_free_beta` | `0.0` | `0.88` |
| `recipe_overrides.betas` | `[0.4, 0.999]` | `[0.0, 0.999]` |
| `recipe_overrides.lr` | `0.007` | `0.014` |
| `recipe_overrides.lr_anneal_start` | `0.55` | _unset_ |
| `recipe_overrides.lr_decay_rate` | _unset_ | `0.55` |
| `recipe_overrides.lr_decay_steps` | _unset_ | `100` |
| `recipe_overrides.lr_floor` | `0.06` | _unset_ |
| `recipe_overrides.lr_schedule` | `"cosine"` | `"exponential"` |
| `recipe_overrides.network_lr_floor` | `0.06` | _unset_ |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.002,
    "decay": 0.9,
    "dim": 16,
    "key_gain": 0.5,
    "kind": "fast_weight",
    "normalize_key": false,
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "advantage_scale": 0.0,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 1.0,
      "assoc_ridge": 0.1,
      "cancel_linear_gan": true,
      "context_bypass_weight": 5.0,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "covariance_ema": 0.05,
      "covariance_eps": 0.0001,
      "feature_scale": 2.1,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "kernel_bandwidth": 0.35,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 100.0,
      "residual_bandwidth": 1.15,
      "residual_kernel": "matern32_krr",
      "residual_ridge": 0.06,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "schedule_free_beta": 0.88,
      "schedule_free_c_min": 0.02,
      "schedule_free_poly_k": 4.0,
      "spike_slab_scale": 1.0,
      "style_bonus_weight": 0.002,
      "subspace_abs_tol": 0.008,
      "subspace_dims": 6,
      "subspace_nu0": 0.1,
      "subspace_rel_tol": 0.12,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0,
      "variance_ratio_scale": 0.15,
      "visit_shrinkage": 0.25
    },
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "d_lr_mult": 0.02,
    "loss": "hinge",
    "lr": 0.014,
    "lr_decay_rate": 0.55,
    "lr_decay_steps": 100,
    "lr_schedule": "exponential",
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 5/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `2e1c3ffe405a476e8406c6379d8a4fe7c58d0b5b6fe4962be9448d8942636544`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-schedule-free-required"></a>
### Required evidence

<a id="online-gail-bcap-schedule-free-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_img_round_robin_bars4.gif) | 2026-11-14T03:41:42.599+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_img_dwell_bars4.gif) | 2026-11-14T03:41:55.790+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.009013`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_img_translating_bar_next.gif) | 2026-11-14T03:42:15.448+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_img_fork_rollout.gif) | 2026-11-14T03:42:35.603+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_img_variable_dwell_bars4.gif) | 2026-11-14T03:42:49.409+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_img_rare_mode_bars4.gif) | 2026-11-14T03:43:03.036+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.01724` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_img_delayed_cue.gif) | 2026-11-14T03:43:15.748+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.09375`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_img_persistent_fork.gif) | 2026-11-14T03:43:46.992+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.024931`; `xy_rmse=0.007993`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_img_circle_orbit.gif) | 2026-11-14T03:44:22.255+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.023827`; `s2c_rmse=0.023791`; `c2s_rmse=0.023862`; `xy_rmse=0.008786`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_img_square_circle_cycle.gif) | 2026-11-14T03:45:33.854+00:00 |

<a id="online-gail-bcap-schedule-free-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-schedule-free-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_minimal_dense4.gif) | 2026-11-14T03:45:37.642+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_minimal_tokens4.gif) | 2026-11-14T03:45:41.679+00:00 |

<a id="online-gail-bcap-schedule-free-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.031152`; `seen_max_abs_error=0.12757`; `heldout_response_rmse=0.02844`; `heldout_max_abs_error=0.072941`; failed terminal bounds: heldout_max_abs_error <= 0.05; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_minimal_kernel2x2.gif) | 2026-11-14T03:45:49.536+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **PASS** | 5/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.99368`; `heldout_exact_accuracy=1`; `heldout_min_bit_confidence=0.9938` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_minimal_addition2bit.gif) | 2026-11-14T03:45:57.571+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **PASS** | 5/5 | `position_delta_rmse_m=0.0017899`; `relative_mse=0.0114`; `moving_direction_accuracy=0.95833`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_robotics_fetch_reach.gif) | 2026-11-14T03:46:15.412+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010548`; `relative_mse=0.74138`; `moving_direction_accuracy=0.69767`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_robotics_fetch_push.gif) | 2026-11-14T03:46:33.314+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0064721`; `relative_mse=0.25496`; `moving_direction_accuracy=0.86275`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_robotics_fetch_slide.gif) | 2026-11-14T03:46:51.689+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **PASS** | 5/5 | `position_delta_rmse_m=0.002256`; `relative_mse=0.032427`; `moving_direction_accuracy=0.78947`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_robotics_fetch_pick_and_place.gif) | 2026-11-14T03:47:10.892+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0083548`; `relative_mse=1.0325`; `moving_direction_accuracy=0.66822`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_robotics_hand_reach.gif) | 2026-11-14T03:47:28.679+00:00 |

<a id="online-gail-bcap-schedule-free-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.24602`; `interpolation_speed_error=0.18841`; `interpolation_raw_rmse=0.21804`; `interpolation_point_validity=0.27083`; `interpolation_direction_accuracy=0.60417`; `extrapolation_centroid_rmse=0.30081`; `extrapolation_speed_error=0.46714`; `extrapolation_raw_rmse=0.083446`; `extrapolation_point_validity=0.20833`; `extrapolation_direction_accuracy=0.58333`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_raw_rmse <= 0.12; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-schedule-free/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-schedule-free_seq_minimal_velocity_extrapolation.gif) | 2026-11-14T03:47:49.399+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-schedule-free --device cuda`.

<a id="online-gail-bcap-subspace"></a>
## `online-gail-bcap-subspace`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-subspace.json)

Parent: [`online-gail-bcap-snr`](gail-bcap.md#online-gail-bcap-snr).

Declared changes:

- Precision-whitened anchored Gram-Schmidt innovation subspace (subspace_dims=6, abs_tol=0.008, rel_tol=0.12, nu_0=0.10, feature_scale=2.1, feature_tolerance=0.15, spike_slab_scale=1.0) plus 2-row deterministic JL tail eliminates blind 8x64 projection nullspace and 1:1 MoG Wiener attenuation without retroactive coordinate drift
- Semiparametric float64 Matern-3/2 kernel ridge regression residual (residual_kernel=matern32_krr, residual_bandwidth=1.15, residual_ridge=0.06) over unquantized per-bucket latent means gated by (1 - g_disc) eliminates cluster-density bias on continuous contact manifolds
- Fine-salience causal reader (contrast_threshold=0.002, write_lr=0.35) and calmed Hinge-BCAP posterior optimization (d_lr_mult=0.02, reconstruction_weight=100.0, style_bonus_weight=0.002, betas=[0.4, 0.999], lr=0.007, lr_anneal_start=0.55, lr_floor=0.06)

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.contrast_threshold` | `0.005` | `0.002` |
| `memory_policy.transport.feature_scale` | `4.0` | `2.1` |
| `memory_policy.transport.feature_tolerance` | `0.3` | `0.15` |
| `memory_policy.transport.reconstruction_weight` | `10.0` | `100.0` |
| `memory_policy.transport.residual_bandwidth` | _unset_ | `1.15` |
| `memory_policy.transport.residual_kernel` | _unset_ | `"matern32_krr"` |
| `memory_policy.transport.residual_ridge` | _unset_ | `0.06` |
| `memory_policy.transport.schedule_free_beta` | _unset_ | `0.0` |
| `memory_policy.transport.schedule_free_c_min` | _unset_ | `0.02` |
| `memory_policy.transport.schedule_free_poly_k` | _unset_ | `4.0` |
| `memory_policy.transport.spike_slab_scale` | `2.0` | `1.0` |
| `memory_policy.transport.style_bonus_weight` | `0.05` | `0.002` |
| `memory_policy.transport.subspace_abs_tol` | _unset_ | `0.008` |
| `memory_policy.transport.subspace_dims` | _unset_ | `6` |
| `memory_policy.transport.subspace_nu0` | _unset_ | `0.1` |
| `memory_policy.transport.subspace_rel_tol` | _unset_ | `0.12` |
| `memory_policy.write_lr` | `0.5` | `0.35` |
| `recipe_overrides.betas` | `[0.0, 0.999]` | `[0.4, 0.999]` |
| `recipe_overrides.d_lr_mult` | _unset_ | `0.02` |
| `recipe_overrides.lr` | `0.005` | `0.007` |
| `recipe_overrides.lr_anneal_start` | `0.5` | `0.55` |
| `recipe_overrides.lr_floor` | `0.05` | `0.06` |
| `recipe_overrides.network_lr_floor` | `0.05` | `0.06` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.002,
    "decay": 0.9,
    "dim": 16,
    "key_gain": 0.5,
    "kind": "fast_weight",
    "normalize_key": false,
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "advantage_scale": 0.0,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 1.0,
      "assoc_ridge": 0.1,
      "cancel_linear_gan": true,
      "context_bypass_weight": 5.0,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "covariance_ema": 0.05,
      "covariance_eps": 0.0001,
      "feature_scale": 2.1,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "kernel_bandwidth": 0.35,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 100.0,
      "residual_bandwidth": 1.15,
      "residual_kernel": "matern32_krr",
      "residual_ridge": 0.06,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "schedule_free_beta": 0.0,
      "schedule_free_c_min": 0.02,
      "schedule_free_poly_k": 4.0,
      "spike_slab_scale": 1.0,
      "style_bonus_weight": 0.002,
      "subspace_abs_tol": 0.008,
      "subspace_dims": 6,
      "subspace_nu0": 0.1,
      "subspace_rel_tol": 0.12,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0,
      "variance_ratio_scale": 0.15,
      "visit_shrinkage": 0.25
    },
    "write_lr": 0.35
  },
  "recipe_overrides": {
    "betas": [
      0.4,
      0.999
    ],
    "d_lr_mult": 0.02,
    "loss": "hinge",
    "lr": 0.007,
    "lr_anneal_start": 0.55,
    "lr_floor": 0.06,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.06,
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 4/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `d96fb63fec9c8c74e8f183b6682a366630ca5b3e57f40634841f672d62f681f2`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-subspace-required"></a>
### Required evidence

<a id="online-gail-bcap-subspace-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_img_round_robin_bars4.gif) | 2026-11-14T03:58:16.913+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_img_dwell_bars4.gif) | 2026-11-14T03:58:24.848+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.008983`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_img_translating_bar_next.gif) | 2026-11-14T03:58:36.303+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_img_fork_rollout.gif) | 2026-11-14T03:58:47.811+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_img_variable_dwell_bars4.gif) | 2026-11-14T03:58:55.850+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_img_rare_mode_bars4.gif) | 2026-11-14T03:59:03.941+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.01488` | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_img_delayed_cue.gif) | 2026-11-14T03:59:15.350+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.09375`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_img_persistent_fork.gif) | 2026-11-14T03:59:46.000+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.014614`; `xy_rmse=0.003566`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_img_circle_orbit.gif) | 2026-11-14T04:00:21.805+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.016947`; `s2c_rmse=0.01592`; `c2s_rmse=0.017974`; `xy_rmse=0.005291`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_img_square_circle_cycle.gif) | 2026-11-14T04:01:29.753+00:00 |

<a id="online-gail-bcap-subspace-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-subspace-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_minimal_dense4.gif) | 2026-11-14T04:01:33.391+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_minimal_tokens4.gif) | 2026-11-14T04:01:37.240+00:00 |

<a id="online-gail-bcap-subspace-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.040107`; `seen_max_abs_error=0.11638`; `heldout_response_rmse=0.046355`; `heldout_max_abs_error=0.11693`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_minimal_kernel2x2.gif) | 2026-11-14T04:01:44.819+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **PASS** | 5/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.98511`; `heldout_exact_accuracy=1`; `heldout_min_bit_confidence=0.98533` | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_minimal_addition2bit.gif) | 2026-11-14T04:01:52.526+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **PASS** | 5/5 | `position_delta_rmse_m=0.0017877`; `relative_mse=0.011372`; `moving_direction_accuracy=0.95833`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_robotics_fetch_reach.gif) | 2026-11-14T04:02:09.588+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011636`; `relative_mse=0.90211`; `moving_direction_accuracy=0.74419`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_robotics_fetch_push.gif) | 2026-11-14T04:02:27.239+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0055822`; `relative_mse=0.18966`; `moving_direction_accuracy=0.88235`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_robotics_fetch_slide.gif) | 2026-11-14T04:02:45.427+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **PASS** | 5/5 | `position_delta_rmse_m=0.0022812`; `relative_mse=0.033153`; `moving_direction_accuracy=0.75439`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_robotics_fetch_pick_and_place.gif) | 2026-11-14T04:03:06.540+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0081517`; `relative_mse=0.98289`; `moving_direction_accuracy=0.64953`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_robotics_hand_reach.gif) | 2026-11-14T04:03:23.689+00:00 |

<a id="online-gail-bcap-subspace-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.24674`; `interpolation_speed_error=0.17978`; `interpolation_raw_rmse=0.19548`; `interpolation_point_validity=0.22917`; `interpolation_direction_accuracy=0.60417`; `extrapolation_centroid_rmse=0.33368`; `extrapolation_speed_error=0.54564`; `extrapolation_raw_rmse=0.091077`; `extrapolation_point_validity=0.10417`; `extrapolation_direction_accuracy=0.54167`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_raw_rmse <= 0.12; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-subspace/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-subspace_seq_minimal_velocity_extrapolation.gif) | 2026-11-14T04:03:43.673+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-subspace --device cuda`.

<a id="online-gail-bcap-snr"></a>
## `online-gail-bcap-snr`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-snr.json)

Parent: [`online-gail-bcap-spike-slab`](gail-bcap.md#online-gail-bcap-spike-slab).

Declared changes:

- Lower causal salience contrast gate (contrast_threshold=0.005) so sparse 4-of-64-pixel precision robotics actions are retained near contact while exact zero-contrast distractors stay gated
- Scale posterior feature projection by 2x (feature_scale=4.0, feature_tolerance=0.30, spike_slab_scale=2.0) to boost latent signal-to-noise ratio by 4x relative to task-owned prior_sigma=0.025, and fix episode-invariant imitation style gain (advantage_scale=0.0)

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.contrast_threshold` | `0.03` | `0.005` |
| `memory_policy.transport.advantage_scale` | `0.25` | `0.0` |
| `memory_policy.transport.feature_scale` | `2.0` | `4.0` |
| `memory_policy.transport.feature_tolerance` | `0.15` | `0.3` |
| `memory_policy.transport.spike_slab_scale` | `1.0` | `2.0` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.005,
    "decay": 0.9,
    "dim": 16,
    "key_gain": 0.5,
    "kind": "fast_weight",
    "normalize_key": false,
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "advantage_scale": 0.0,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 1.0,
      "assoc_ridge": 0.1,
      "cancel_linear_gan": true,
      "context_bypass_weight": 5.0,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "covariance_ema": 0.05,
      "covariance_eps": 0.0001,
      "feature_scale": 4.0,
      "feature_tolerance": 0.3,
      "image_center": 0.5,
      "kernel_bandwidth": 0.35,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 10.0,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "spike_slab_scale": 2.0,
      "style_bonus_weight": 0.05,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0,
      "variance_ratio_scale": 0.15,
      "visit_shrinkage": 0.25
    },
    "write_lr": 0.5
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "loss": "hinge",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 3/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `50ab4ca17b2b680a1436cd54833b15defa85c64c55449060db323306ac152727`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-snr-required"></a>
### Required evidence

<a id="online-gail-bcap-snr-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_round_robin_bars4.gif) | 2026-11-14T03:47:58.610+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_dwell_bars4.gif) | 2026-11-14T03:48:06.567+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.008297`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_translating_bar_next.gif) | 2026-11-14T03:48:17.351+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_fork_rollout.gif) | 2026-11-14T03:48:28.184+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_variable_dwell_bars4.gif) | 2026-11-14T03:48:36.013+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_rare_mode_bars4.gif) | 2026-11-14T03:48:43.984+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.022543` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_delayed_cue.gif) | 2026-11-14T03:48:54.782+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.09375`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_persistent_fork.gif) | 2026-11-14T03:49:22.371+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.01578`; `xy_rmse=0.003225`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_circle_orbit.gif) | 2026-11-14T03:49:54.495+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.018057`; `s2c_rmse=0.017729`; `c2s_rmse=0.018384`; `xy_rmse=0.006028`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_img_square_circle_cycle.gif) | 2026-11-14T03:50:55.324+00:00 |

<a id="online-gail-bcap-snr-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-snr-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_minimal_dense4.gif) | 2026-11-14T03:50:58.927+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_minimal_tokens4.gif) | 2026-11-14T03:51:02.738+00:00 |

<a id="online-gail-bcap-snr-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.042176`; `seen_max_abs_error=0.14914`; `heldout_response_rmse=0.040025`; `heldout_max_abs_error=0.093714`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_minimal_kernel2x2.gif) | 2026-11-14T03:51:10.137+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **PASS** | 5/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.96137`; `heldout_exact_accuracy=1`; `heldout_min_bit_confidence=0.96087` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_minimal_addition2bit.gif) | 2026-11-14T03:51:17.610+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **PASS** | 5/5 | `position_delta_rmse_m=0.0019153`; `relative_mse=0.013053`; `moving_direction_accuracy=0.97917`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_robotics_fetch_reach.gif) | 2026-11-14T03:51:33.179+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010593`; `relative_mse=0.74765`; `moving_direction_accuracy=0.62791`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_robotics_fetch_push.gif) | 2026-11-14T03:51:49.562+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0069162`; `relative_mse=0.29114`; `moving_direction_accuracy=0.80392`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_robotics_fetch_slide.gif) | 2026-11-14T03:52:06.983+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 4/5 | `position_delta_rmse_m=0.0024542`; `relative_mse=0.038372`; `moving_direction_accuracy=0.78947`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_robotics_fetch_pick_and_place.gif) | 2026-11-14T03:52:22.803+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0078394`; `relative_mse=0.90903`; `moving_direction_accuracy=0.66822`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_robotics_hand_reach.gif) | 2026-11-14T03:52:38.299+00:00 |

<a id="online-gail-bcap-snr-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.046327`; `interpolation_speed_error=0.050918`; `interpolation_raw_rmse=0.036157`; `interpolation_point_validity=0.16667`; `interpolation_direction_accuracy=0.91667`; `extrapolation_centroid_rmse=0.36643`; `extrapolation_speed_error=0.4341`; `extrapolation_raw_rmse=0.081088`; `extrapolation_point_validity=0.25`; `extrapolation_direction_accuracy=0.66667`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_raw_rmse <= 0.12; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-snr/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-snr_seq_minimal_velocity_extrapolation.gif) | 2026-11-14T03:52:57.826+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-snr --device cuda`.

<a id="online-gail-bcap-spike-slab"></a>
## `online-gail-bcap-spike-slab`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-spike-slab.json)

Parent: [`online-gail-bcap-bce`](gail-bcap.md#online-gail-bcap-bce).

Declared changes:

- Replace radial distance blending with float64 Sherman-Morrison RLS (assoc_ridge=0.1) + precision-weighted Mahalanobis spike-and-slab prototype transport (spike_slab_scale=1.0) and empirical-Bayes kernel-gated active-entropy residual quantiles
- Add Markov context-bypass regularizer (context_bypass_weight=5.0), calibrated sigmoid style bonus (style_bonus_weight=0.05), and unnormalized linear causal key projection (normalize_key=false, key_gain=0.5) preserving Euclidean superposition norm

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.key_gain` | _unset_ | `0.5` |
| `memory_policy.normalize_key` | _unset_ | `false` |
| `memory_policy.transport.assoc_ridge` | `0.05` | `0.1` |
| `memory_policy.transport.context_bypass_weight` | _unset_ | `5.0` |
| `memory_policy.transport.covariance_ema` | _unset_ | `0.05` |
| `memory_policy.transport.covariance_eps` | _unset_ | `0.0001` |
| `memory_policy.transport.kernel_bandwidth` | _unset_ | `0.35` |
| `memory_policy.transport.spike_slab_scale` | _unset_ | `1.0` |
| `memory_policy.transport.style_bonus_weight` | `0.1` | `0.05` |
| `memory_policy.transport.variance_ratio_scale` | _unset_ | `0.15` |
| `memory_policy.transport.visit_shrinkage` | _unset_ | `0.25` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.03,
    "decay": 0.9,
    "dim": 16,
    "key_gain": 0.5,
    "kind": "fast_weight",
    "normalize_key": false,
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "advantage_scale": 0.25,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 1.0,
      "assoc_ridge": 0.1,
      "cancel_linear_gan": true,
      "context_bypass_weight": 5.0,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "covariance_ema": 0.05,
      "covariance_eps": 0.0001,
      "feature_scale": 2.0,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "kernel_bandwidth": 0.35,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 10.0,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "spike_slab_scale": 1.0,
      "style_bonus_weight": 0.05,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0,
      "variance_ratio_scale": 0.15,
      "visit_shrinkage": 0.25
    },
    "write_lr": 0.5
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "loss": "hinge",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 3/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `bb4ebe57e0f98a9054030c0e09f83523f6a79c2856e10f6472ec0fe6a6308d3a`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-spike-slab-required"></a>
### Required evidence

<a id="online-gail-bcap-spike-slab-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_round_robin_bars4.gif) | 2026-11-14T03:53:07.266+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_dwell_bars4.gif) | 2026-11-14T03:53:15.149+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.009513`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_translating_bar_next.gif) | 2026-11-14T03:53:26.101+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_fork_rollout.gif) | 2026-11-14T03:53:36.836+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_variable_dwell_bars4.gif) | 2026-11-14T03:53:44.854+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_rare_mode_bars4.gif) | 2026-11-14T03:53:52.767+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.027101` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_delayed_cue.gif) | 2026-11-14T03:54:03.734+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.09375`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_persistent_fork.gif) | 2026-11-14T03:54:31.611+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.019609`; `xy_rmse=0.005184`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_circle_orbit.gif) | 2026-11-14T03:55:03.617+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.019104`; `s2c_rmse=0.018582`; `c2s_rmse=0.019626`; `xy_rmse=0.005748`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_img_square_circle_cycle.gif) | 2026-11-14T03:56:03.619+00:00 |

<a id="online-gail-bcap-spike-slab-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-spike-slab-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_minimal_dense4.gif) | 2026-11-14T03:56:07.229+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_minimal_tokens4.gif) | 2026-11-14T03:56:11.053+00:00 |

<a id="online-gail-bcap-spike-slab-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.041768`; `seen_max_abs_error=0.18007`; `heldout_response_rmse=0.035511`; `heldout_max_abs_error=0.083195`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_minimal_kernel2x2.gif) | 2026-11-14T03:56:18.520+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **PASS** | 5/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.95429`; `heldout_exact_accuracy=1`; `heldout_min_bit_confidence=0.9533` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_minimal_addition2bit.gif) | 2026-11-14T03:56:26.164+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **PASS** | 5/5 | `position_delta_rmse_m=0.0034835`; `relative_mse=0.043181`; `moving_direction_accuracy=0.95833`; `finite_fraction=1` | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_robotics_fetch_reach.gif) | 2026-11-14T03:56:41.636+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010513`; `relative_mse=0.73647`; `moving_direction_accuracy=0.69767`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_robotics_fetch_push.gif) | 2026-11-14T03:56:58.807+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0074842`; `relative_mse=0.34093`; `moving_direction_accuracy=0.76471`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_robotics_fetch_slide.gif) | 2026-11-14T03:57:16.038+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 2/5 | `position_delta_rmse_m=0.0024022`; `relative_mse=0.036766`; `moving_direction_accuracy=0.73684`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_robotics_fetch_pick_and_place.gif) | 2026-11-14T03:57:32.303+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0082151`; `relative_mse=0.99823`; `moving_direction_accuracy=0.61682`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_robotics_hand_reach.gif) | 2026-11-14T03:57:47.631+00:00 |

<a id="online-gail-bcap-spike-slab-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.051051`; `interpolation_speed_error=0.056314`; `interpolation_raw_rmse=0.036744`; `interpolation_point_validity=0`; `interpolation_direction_accuracy=0.83333`; `extrapolation_centroid_rmse=0.26227`; `extrapolation_speed_error=0.24693`; `extrapolation_raw_rmse=0.080967`; `extrapolation_point_validity=0.16667`; `extrapolation_direction_accuracy=0.66667`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_direction_accuracy == 1.0; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-spike-slab/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-spike-slab_seq_minimal_velocity_extrapolation.gif) | 2026-11-14T03:58:07.349+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-spike-slab --device cuda`.

<a id="online-gail-bcap-bce"></a>
## `online-gail-bcap-bce`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-bce.json)

Parent: [`online-gail-bcap-hinge`](gail-bcap.md#online-gail-bcap-hinge).

Declared changes:

- Enable detached Sherman-Morrison recursive associative fast-weight operator (assoc_ridge=0.05, assoc_forgetting=1.0, assoc_blend_scale=0.25, assoc_max_blend=1.0) to extrapolate latent feature expectations smoothly beyond discrete context tolerance

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.transport.assoc_max_blend` | `0.0` | `1.0` |
| `memory_policy.transport.assoc_ridge` | `0.0` | `0.05` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.03,
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "advantage_scale": 0.25,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 1.0,
      "assoc_ridge": 0.05,
      "cancel_linear_gan": true,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "feature_scale": 2.0,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 10.0,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "style_bonus_weight": 0.1,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0
    },
    "write_lr": 0.5
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "loss": "hinge",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 1/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `010caed96830d115255257866a111e9acaa9a94c950a4fa54bfcf0510dc48592`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-bce-required"></a>
### Required evidence

<a id="online-gail-bcap-bce-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_round_robin_bars4.gif) | 2026-11-14T03:02:19.287+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_dwell_bars4.gif) | 2026-11-14T03:02:27.448+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.006364`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_translating_bar_next.gif) | 2026-11-14T03:02:36.548+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_fork_rollout.gif) | 2026-11-14T03:02:45.879+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_variable_dwell_bars4.gif) | 2026-11-14T03:02:53.893+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_rare_mode_bars4.gif) | 2026-11-14T03:03:02.058+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.023147` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_delayed_cue.gif) | 2026-11-14T03:03:11.201+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.03125`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_persistent_fork.gif) | 2026-11-14T03:03:30.129+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.011924`; `xy_rmse=0.003893`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_circle_orbit.gif) | 2026-11-14T03:03:51.208+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.012503`; `s2c_rmse=0.011269`; `c2s_rmse=0.013737`; `xy_rmse=0.003717`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_img_square_circle_cycle.gif) | 2026-11-14T03:04:29.945+00:00 |

<a id="online-gail-bcap-bce-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-bce-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_dense4.gif) | 2026-11-14T03:04:33.517+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_tokens4.gif) | 2026-11-14T03:04:37.351+00:00 |

<a id="online-gail-bcap-bce-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.04189`; `seen_max_abs_error=0.15896`; `heldout_response_rmse=0.086803`; `heldout_max_abs_error=0.18952`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_kernel2x2.gif) | 2026-11-14T03:04:43.121+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.97076`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=1.1497e-08`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_addition2bit.gif) | 2026-11-14T03:04:49.031+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0053724`; `relative_mse=0.1027`; `moving_direction_accuracy=0.95833`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_fetch_reach.gif) | 2026-11-14T03:05:02.147+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.011217`; `relative_mse=0.83837`; `moving_direction_accuracy=0.74419`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_fetch_push.gif) | 2026-11-14T03:05:15.432+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0077145`; `relative_mse=0.36224`; `moving_direction_accuracy=0.80392`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_fetch_slide.gif) | 2026-11-14T03:05:28.637+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 4/5 | `position_delta_rmse_m=0.0025834`; `relative_mse=0.04252`; `moving_direction_accuracy=0.80702`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_fetch_pick_and_place.gif) | 2026-11-14T03:05:41.987+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0081327`; `relative_mse=0.97832`; `moving_direction_accuracy=0.58411`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_robotics_hand_reach.gif) | 2026-11-14T03:05:55.227+00:00 |

<a id="online-gail-bcap-bce-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.035795`; `interpolation_speed_error=0.04004`; `interpolation_raw_rmse=0.029099`; `interpolation_point_validity=0.75`; `interpolation_direction_accuracy=0.91667`; `extrapolation_centroid_rmse=0.077977`; `extrapolation_speed_error=0.060972`; `extrapolation_raw_rmse=0.050648`; `extrapolation_point_validity=0.5`; `extrapolation_direction_accuracy=1`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-bce/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-bce_seq_minimal_velocity_extrapolation.gif) | 2026-11-14T03:06:09.931+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-bce --device cuda`.

<a id="online-gail-bcap-hinge"></a>
## `online-gail-bcap-hinge`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-hinge.json)

Parent: [`online-feature-prior-bce`](featureprior-bcap.md#online-feature-prior-bce).

Declared changes:

- Switch critic adversarial objective to Hinge loss paired with BCAP input-gradient cap and write_lr=0.5 causal-state fast-weight updates
- Stabilized GAIL-BCAP generator objective: cancel unbounded linear Hinge generator slope and optimize prioritized BCE survival anchor minus bounded sigmoid critic style bonus (weight 0.1) modulated by causal discounted return advantage (gamma=0.99, scale=0.25)
- Entropy-regularized categorical outcome quantiles (bonus=0.01) over active observed prototypes per context; associative fast-weight extrapolation disabled (assoc_ridge=0, assoc_max_blend=0)

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `memory_policy.transport.advantage_scale` | _unset_ | `0.25` |
| `memory_policy.transport.assoc_blend_scale` | _unset_ | `0.25` |
| `memory_policy.transport.assoc_forgetting` | _unset_ | `1.0` |
| `memory_policy.transport.assoc_max_blend` | _unset_ | `0.0` |
| `memory_policy.transport.assoc_ridge` | _unset_ | `0.0` |
| `memory_policy.transport.cancel_linear_gan` | _unset_ | `true` |
| `memory_policy.transport.policy_entropy_bonus` | _unset_ | `0.01` |
| `memory_policy.transport.return_discount` | _unset_ | `0.99` |
| `memory_policy.transport.router` | `"feature_conditional"` | `"gail_conditional"` |
| `memory_policy.transport.style_bonus_weight` | _unset_ | `0.1` |
| `memory_policy.write_lr` | `0.35` | `0.5` |
| `recipe_overrides.loss` | `"non_saturating"` | `"hinge"` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.03,
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "advantage_scale": 0.25,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 0.0,
      "assoc_ridge": 0.0,
      "cancel_linear_gan": true,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "feature_scale": 2.0,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 10.0,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "style_bonus_weight": 0.1,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0
    },
    "write_lr": 0.5
  },
  "recipe_overrides": {
    "betas": [
      0.0,
      0.999
    ],
    "loss": "hinge",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 1/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `1bce0bff0efab1534109b49a04efa41ef80284b4489bb6ab68d3037476cf84bf`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-hinge-required"></a>
### Required evidence

<a id="online-gail-bcap-hinge-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_round_robin_bars4.gif) | 2026-11-14T03:23:19.242+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_dwell_bars4.gif) | 2026-11-14T03:23:27.100+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.007497`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_translating_bar_next.gif) | 2026-11-14T03:23:35.962+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_fork_rollout.gif) | 2026-11-14T03:23:44.767+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_variable_dwell_bars4.gif) | 2026-11-14T03:23:52.767+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_rare_mode_bars4.gif) | 2026-11-14T03:24:00.705+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.023308` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_delayed_cue.gif) | 2026-11-14T03:24:09.705+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.03125`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_persistent_fork.gif) | 2026-11-14T03:24:27.513+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.011981`; `xy_rmse=0.003659`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_circle_orbit.gif) | 2026-11-14T03:24:47.938+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.011578`; `s2c_rmse=0.01075`; `c2s_rmse=0.012407`; `xy_rmse=0.00359`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_img_square_circle_cycle.gif) | 2026-11-14T03:25:25.053+00:00 |

<a id="online-gail-bcap-hinge-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-hinge-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_dense4.gif) | 2026-11-14T03:25:28.796+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_tokens4.gif) | 2026-11-14T03:25:32.466+00:00 |

<a id="online-gail-bcap-hinge-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.041775`; `seen_max_abs_error=0.16378`; `heldout_response_rmse=0.26537`; `heldout_max_abs_error=0.43112`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_kernel2x2.gif) | 2026-11-14T03:25:38.276+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.96474`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=9.6964e-10`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_addition2bit.gif) | 2026-11-14T03:25:43.938+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.005476`; `relative_mse=0.1067`; `moving_direction_accuracy=0.95833`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_fetch_reach.gif) | 2026-11-14T03:25:56.660+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.01111`; `relative_mse=0.82236`; `moving_direction_accuracy=0.69767`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_fetch_push.gif) | 2026-11-14T03:26:09.358+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0075477`; `relative_mse=0.34674`; `moving_direction_accuracy=0.76471`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_fetch_slide.gif) | 2026-11-14T03:26:22.234+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 1/5 | `position_delta_rmse_m=0.0026558`; `relative_mse=0.044936`; `moving_direction_accuracy=0.73684`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_fetch_pick_and_place.gif) | 2026-11-14T03:26:35.011+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0081538`; `relative_mse=0.9834`; `moving_direction_accuracy=0.58879`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_robotics_hand_reach.gif) | 2026-11-14T03:26:47.825+00:00 |

<a id="online-gail-bcap-hinge-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.031857`; `interpolation_speed_error=0.039085`; `interpolation_raw_rmse=0.026463`; `interpolation_point_validity=0.83333`; `interpolation_direction_accuracy=0.91667`; `extrapolation_centroid_rmse=0.079113`; `extrapolation_speed_error=0.066138`; `extrapolation_raw_rmse=0.05589`; `extrapolation_point_validity=0.5`; `extrapolation_direction_accuracy=1`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-hinge/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-hinge_seq_minimal_velocity_extrapolation.gif) | 2026-11-14T03:27:01.952+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-hinge --device cuda`.

<a id="online-gail-bcap-momentum"></a>
## `online-gail-bcap-momentum`

[Exact declaration](../../../configs/forge/ideas/online-gail-bcap-momentum.json)

Parent: [`online-gail-bcap-bce`](gail-bcap.md#online-gail-bcap-bce).

Declared changes:

- Enable nonzero Adam first-moment smoothing (betas=[0.5, 0.999]) across sequential batch_size=1 arrivals

### Configuration delta from parent

| Field | Parent | This recipe |
| --- | --- | --- |
| `recipe_overrides.betas` | `[0.0, 0.999]` | `[0.5, 0.999]` |

### Effective declaration

```json
{
  "memory_policy": {
    "age_horizon": 8.0,
    "association_weight": 0.0,
    "change_threshold": 0.04,
    "context_scale": 3.0,
    "contrast_threshold": 0.03,
    "decay": 0.9,
    "dim": 16,
    "kind": "fast_weight",
    "reader": "causal_state",
    "rehearsal_batch": 0,
    "replay_policy": "none",
    "reservoir_size": 64,
    "transport": {
      "advantage_scale": 0.25,
      "assoc_blend_scale": 0.25,
      "assoc_forgetting": 1.0,
      "assoc_max_blend": 1.0,
      "assoc_ridge": 0.05,
      "cancel_linear_gan": true,
      "context_capacity": 64,
      "context_clip": 8.0,
      "context_tolerance": 0.25,
      "feature_scale": 2.0,
      "feature_tolerance": 0.15,
      "image_center": 0.5,
      "policy_entropy_bonus": 0.01,
      "reconstruction_loss": "binary_cross_entropy",
      "reconstruction_weight": 10.0,
      "return_discount": 0.99,
      "router": "gail_conditional",
      "style_bonus_weight": 0.1,
      "temperature": 0.0,
      "usage_decay": 1.0,
      "usage_penalty": 0.0
    },
    "write_lr": 0.5
  },
  "recipe_overrides": {
    "betas": [
      0.5,
      0.999
    ],
    "loss": "hinge",
    "lr": 0.005,
    "lr_anneal_start": 0.5,
    "lr_floor": 0.05,
    "lr_schedule": "cosine",
    "network_lr_floor": 0.05,
    "prior_update": "frozen"
  },
  "recipe_preset": "bcap_adam"
}
```

Current `cuda`, seed 0: **PASS**, 10/10 required passes (10 measured), 1/10 stretch passes (10 measured); qualified tier 1.

Receipt identity: candidate_fingerprint `afe55ca27079ef913fe69606c46a3c962c6a9bdfd4a51da438d2be03a7426e0b`; engine_source_sha256 `f9b1bdb63c640ec7f6bd3036bf40e71cbf2bae7cc3cf8019b6629f77ceda2774`. Each receipt also binds its task and adapter source.

<a id="online-gail-bcap-momentum-required"></a>
### Required evidence

<a id="online-gail-bcap-momentum-tier-1-required"></a>
#### Tier 1 required

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_img_round_robin_bars4`](../technique-inventory.md#seq_img_round_robin_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_round_robin_bars4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_round_robin_bars4.gif) | 2026-11-14T03:37:17.774+00:00 |
| [`seq_img_dwell_bars4`](../technique-inventory.md#seq_img_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0.03125` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_dwell_bars4.gif) | 2026-11-14T03:37:25.698+00:00 |
| [`seq_img_translating_bar_next`](../technique-inventory.md#seq_img_translating_bar_next) | **PASS** | 5/5 | `next_frame_rmse=0.00797`; `direction_acc=1` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_translating_bar_next.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_translating_bar_next.gif) | 2026-11-14T03:37:34.754+00:00 |
| [`seq_img_fork_rollout`](../technique-inventory.md#seq_img_fork_rollout) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_fork_rollout.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_fork_rollout.gif) | 2026-11-14T03:37:43.787+00:00 |
| [`seq_img_variable_dwell_bars4`](../technique-inventory.md#seq_img_variable_dwell_bars4) | **PASS** | 5/5 | `hq=1`; `modes=4`; `finite_template_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_variable_dwell_bars4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_variable_dwell_bars4.gif) | 2026-11-14T03:37:51.762+00:00 |
| [`seq_img_rare_mode_bars4`](../technique-inventory.md#seq_img_rare_mode_bars4) | **PASS** | 5/5 | `hq=1`; `finite_template_tv=0`; `support_modes=4`; `rare_mass=0.0625`; `rare_mass=0.0625` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_rare_mode_bars4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_rare_mode_bars4.gif) | 2026-11-14T03:37:59.623+00:00 |
| [`seq_img_delayed_cue`](../technique-inventory.md#seq_img_delayed_cue) | **PASS** | 5/5 | `outcome_accuracy=1`; `prediction_rmse=0.023138` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_delayed_cue.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_delayed_cue.gif) | 2026-11-14T03:38:08.686+00:00 |
| [`seq_img_persistent_fork`](../technique-inventory.md#seq_img_persistent_fork) | **PASS** | 5/5 | `hq=1`; `within_state_modes=2`; `branch_tv=0.03125`; `trajectory_consistency=1` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_persistent_fork.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_persistent_fork.gif) | 2026-11-14T03:38:27.242+00:00 |
| [`seq_img_circle_orbit`](../technique-inventory.md#seq_img_circle_orbit) | **PASS** | 5/5 | `orbit_rmse=0.013045`; `xy_rmse=0.003758`; `step_accuracy=1`; `orbit_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_circle_orbit.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_circle_orbit.gif) | 2026-11-14T03:38:48.589+00:00 |
| [`seq_img_square_circle_cycle`](../technique-inventory.md#seq_img_square_circle_cycle) | **PASS** | 5/5 | `cycle_rmse=0.012296`; `s2c_rmse=0.011309`; `c2s_rmse=0.013282`; `xy_rmse=0.00323`; `step_accuracy=1`; `cycle_completion=1`; `waypoint_modes=8` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_img_square_circle_cycle.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_img_square_circle_cycle.gif) | 2026-11-14T03:39:27.319+00:00 |

<a id="online-gail-bcap-momentum-stretch"></a>
### Stretch evidence

<a id="online-gail-bcap-momentum-tier-1-stretch"></a>
#### Tier 1 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_dense4`](../technique-inventory.md#seq_minimal_dense4) | **FAIL** | 0/5 | `valid_fraction=0`; `modes=0`; `mass_tv=1`; failed terminal bounds: mass_tv <= 0.15; modes == 4.0; valid_fraction >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_dense4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_dense4.gif) | 2026-11-14T03:39:31.075+00:00 |
| [`seq_minimal_tokens4`](../technique-inventory.md#seq_minimal_tokens4) | **PASS** | 5/5 | `valid_fraction=1`; `modes=4`; `mass_tv=0` | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_tokens4.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_tokens4.gif) | 2026-11-14T03:39:34.722+00:00 |

<a id="online-gail-bcap-momentum-tier-2-stretch"></a>
#### Tier 2 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_kernel2x2`](../technique-inventory.md#seq_minimal_kernel2x2) | **FAIL** | 0/5 | `seen_response_rmse=0.0353`; `seen_max_abs_error=0.1779`; `heldout_response_rmse=0.0899`; `heldout_max_abs_error=0.23582`; failed terminal bounds: heldout_max_abs_error <= 0.05; heldout_response_rmse <= 0.03; seen_max_abs_error <= 0.05; seen_response_rmse <= 0.03 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_kernel2x2.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_kernel2x2.gif) | 2026-11-14T03:39:40.719+00:00 |
| [`seq_minimal_addition2bit`](../technique-inventory.md#seq_minimal_addition2bit) | **FAIL** | 0/5 | `seen_exact_accuracy=1`; `seen_min_bit_confidence=0.9756`; `heldout_exact_accuracy=0`; `heldout_min_bit_confidence=8.2579e-08`; failed terminal bounds: heldout_exact_accuracy == 1.0; heldout_min_bit_confidence >= 0.9 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_addition2bit.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_addition2bit.gif) | 2026-11-14T03:39:46.496+00:00 |
| [`seq_robotics_fetch_reach`](../technique-inventory.md#seq_robotics_fetch_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0054013`; `relative_mse=0.10381`; `moving_direction_accuracy=0.95833`; `finite_fraction=1`; failed terminal bounds: position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_fetch_reach.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_fetch_reach.gif) | 2026-11-14T03:39:59.778+00:00 |
| [`seq_robotics_fetch_push`](../technique-inventory.md#seq_robotics_fetch_push) | **FAIL** | 0/5 | `position_delta_rmse_m=0.010969`; `relative_mse=0.80171`; `moving_direction_accuracy=0.67442`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_fetch_push.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_fetch_push.gif) | 2026-11-14T03:40:13.021+00:00 |
| [`seq_robotics_fetch_slide`](../technique-inventory.md#seq_robotics_fetch_slide) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0074193`; `relative_mse=0.33504`; `moving_direction_accuracy=0.7451`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_fetch_slide.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_fetch_slide.gif) | 2026-11-14T03:40:26.357+00:00 |
| [`seq_robotics_fetch_pick_and_place`](../technique-inventory.md#seq_robotics_fetch_pick_and_place) | **FAIL** | 2/5 | `position_delta_rmse_m=0.0028069`; `relative_mse=0.050196`; `moving_direction_accuracy=0.80702`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_fetch_pick_and_place.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_fetch_pick_and_place.gif) | 2026-11-14T03:40:42.391+00:00 |
| [`seq_robotics_hand_reach`](../technique-inventory.md#seq_robotics_hand_reach) | **FAIL** | 0/5 | `position_delta_rmse_m=0.0080597`; `relative_mse=0.96082`; `moving_direction_accuracy=0.6028`; `finite_fraction=1`; failed terminal bounds: moving_direction_accuracy >= 0.75; position_delta_rmse_m <= 0.005; relative_mse <= 0.8 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_robotics_hand_reach.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_robotics_hand_reach.gif) | 2026-11-14T03:41:04.505+00:00 |

<a id="online-gail-bcap-momentum-tier-3-stretch"></a>
#### Tier 3 stretch

| Task and gates | Verdict | Terminal checks | Final gate readings / failure | Evidence | Completed UTC |
| --- | --- | --- | --- | --- | --- |
| [`seq_minimal_velocity_extrapolation`](../technique-inventory.md#seq_minimal_velocity_extrapolation) | **FAIL** | 0/5 | `interpolation_centroid_rmse=0.031232`; `interpolation_speed_error=0.037349`; `interpolation_raw_rmse=0.025589`; `interpolation_point_validity=0.66667`; `interpolation_direction_accuracy=1`; `extrapolation_centroid_rmse=0.073497`; `extrapolation_speed_error=0.03789`; `extrapolation_raw_rmse=0.053371`; `extrapolation_point_validity=0.58333`; `extrapolation_direction_accuracy=1`; failed terminal bounds: extrapolation_centroid_rmse <= 0.01; extrapolation_point_validity == 1.0; extrapolation_speed_error <= 0.01; interpolation_centroid_rmse <= 0.01; interpolation_direction_accuracy == 1.0; interpolation_point_validity == 1.0; interpolation_speed_error <= 0.01 | [receipt](../receipts/cuda/online-gail-bcap-momentum/seq_minimal_velocity_extrapolation.json) · [goal.gif](../gifs/online-gail-bcap-momentum_seq_minimal_velocity_extrapolation.gif) | 2026-11-14T03:41:27.643+00:00 |

Reproduce this exact global recipe on every task: `particlestream compare --all-tasks --candidate online-gail-bcap-momentum --device cuda`.

<a id="refresh"></a>
## Refresh

```sh
particlestream compile --device cuda
```

This page is generated alongside the leaderboard from committed source-bound receipts and declarations; editing a generated page cannot change a verdict or earn qualification.

<a id="references"></a>
## References

- [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476). Imitation-style occupancy matching with causal return advantage and policy entropy regularization.
- [Geometric GAN](https://arxiv.org/abs/1705.02894). Hinge margin critic objective paired with the BCAP input-gradient cap.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). Discounted return baseline and variance-standardized advantage gain for online policy updates.
- [Bayesian Variable Selection in Linear Regression](https://doi.org/10.1080/01621459.1988.10478693). Spike-and-slab Bayes factor formulation powering Mahalanobis discrete-vs-continuous prototype gating.
- [Numerics of Gram-Schmidt Orthogonalization](https://doi.org/10.1016/0024-3795(94)90493-6). Two-pass classical Gram-Schmidt reorthogonalization anchoring precision-whitened online innovation directions.
- [The Road Less Scheduled](https://arxiv.org/abs/2405.15682). Primal-Dual Schedule-Free iterate interpolation (y_t, z_t, x_t) paired with causal horizon-invariant step cooling.
