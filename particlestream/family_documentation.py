"""Maintained descriptive family documentation; never a scientific grading input."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from urllib.parse import urlsplit


TRAINING_DETAILS = (
    ("loss", "Adversarial loss"),
    ("optimizer", "Optimizer"),
    ("learning_rates", "Learning rates and annealing"),
    ("gradient_clipping", "Parameter-gradient clipping"),
    ("critic_penalty", "Critic penalties and anchors"),
    ("damping_and_guards", "Damping and update guards"),
    ("online_memory_and_routing", "Online memory, prior and routing"),
    ("noise", "Training and sampling noise"),
    ("averaging", "Parameter averaging and serving"),
)


FAMILY_DOCUMENTATION = {
    "SubspacePrior-BCAP": {
        "overview": (
            "SubspacePrior-BCAP extends ResidualPrior-BCAP (`SubspaceResidualConditionalReconstruction`) to "
            "eliminate two coupled bottlenecks of fixed-projection online priors under strict `batch_size=1` "
            "streaming (`replay_policy=\"none\"`): (1) blind `8×64` random-projection nullspace loss and 1:1 MoG "
            "Wiener attenuation on millimetre physical displacements, solved by causally anchoring up to "
            "$K_{\\mathrm{sub}}=6$ orthonormal two-pass Gram-Schmidt innovation directions with fixed discovery-time "
            "precision-whitening gains alongside a 2-row deterministic Johnson-Lindenstrauss tail; and (2) global "
            "hyperplane collapse on piecewise-nonlinear continuous dynamics, solved by recording exact unquantized "
            "per-bucket latent means and adding a smooth Nadaraya-Watson kernel-ridge residual gated by $(1-g_{\\mathrm{disc},t})$."
        ),
        "symbols": (
            "$G$ is the task-owned residual generator with input layer $W_{\\mathrm{fc}}=[W_z,\\,W_{\\mathrm{ctx}}]$, "
            "$D$ the raw-score convolutional critic, and $P$ the frozen $K$-row `MoGParticlePrior` ($K=32$, $d_z=8$, "
            "$\\sigma=0.025$) with base centers $\\mu_i^{\\mathrm{base}}\\in\\mathbb R^{d_z}$. At step $t$, "
            "$x_t\\in[0,1]^{1\\times 8\\times 8}$ is the single arriving frame, $v_t=\\bar x_t-c_0\\mathbf 1_{64}\\in\\mathbb R^{64}$ "
            "($c_0=0.5$), $E_0\\in\\mathbb R^{d_z\\times 64}$ is the deterministic orthogonal fallback matrix, "
            "$K_{\\mathrm{sub}}=6$ is the anchored subspace capacity, $u_j\\in\\mathbb R^{64}$ and $g_j>0$ ($0\\le j<k_t\\le K_{\\mathrm{sub}}$) "
            "are committed unit innovation basis vectors and fixed precision-whitening gains ($\\tau_{\\mathrm{abs}}=0.008$, "
            "$\\tau_{\\mathrm{rel}}=0.12$, $\\nu_0=0.10$, $s_f=2.1$). $c_{t-1}\\in\\mathbb R^{d_c}$ ($d_c=16$) is the "
            "detached causal context ($\\tau_{\\mathrm{con}}\\in\\{0.03,0.002\\}$), $(W_t,P_t,S_t,q_t)$ are online `float64` "
            "RLS and residual-covariance statistics ($\\lambda_{\\mathrm{rls}}=0.1$), $g_{\\mathrm{disc},t}\\in(0,1]$ is the "
            "empirical-Bayes discreteness gate, and $(k_b^{\\mathrm{ctx}},V_b,\\bar z_b)$ are context keys, visit counts, "
            "and exact unquantized latent means ($C_{\\max}=64$, $\\omega_{\\mathrm{nw}}=1.0$, $h_{\\mathrm{nw}}=0.65$, $\\kappa_{\\mathrm{nw}}=0.06$)."
        ),
        "equations": [
            {
                "label": "Precision-Whitened Anchored Gram-Schmidt Innovation Subspace and JL Tail",
                "latex": (
                    "\\begin{aligned}"
                    "r_t&=\\Big(I-U_{k_{t-1}}U_{k_{t-1}}^\\top\\Big)^2 v_t,\\qquad "
                    "\\mathcal I_t=\\mathbb I\\!\\left[k_{t-1}<K_{\\mathrm{sub}}\\;\\wedge\\;\\lVert r_t\\rVert_2>\\tau_{\\mathrm{abs}}\\;"
                    "\\wedge\\;\\frac{\\lVert r_t\\rVert_2}{\\lVert v_t\\rVert_2+10^{-3}}>\\tau_{\\mathrm{rel}}\\right],\\\\"
                    "u_{k_{t-1}}&=\\frac{r_t}{\\lVert r_t\\rVert_2},\\qquad "
                    "g_{k_{t-1}}=\\big(\\lVert r_t\\rVert_2^2+\\nu_0^2\\big)^{-1/2}\\quad(\\text{anchored once when }\\mathcal I_t=1),\\\\"
                    "\\phi_t(x)&=s_f\\begin{bmatrix}\\operatorname{diag}(g_{0:k_t-1})U_{k_t}^\\top\\\\0_{(K_{\\mathrm{sub}}-k_t)\\times 64}\\\\"
                    "E_{0,\\,K_{\\mathrm{sub}}:d_z}\\end{bmatrix}(\\bar x-c_0\\mathbf 1_{64})."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Two-pass classical Gram-Schmidt orthogonalization in `float64` captures exact active target "
                    "subspaces while the dimensionless relative innovation gate $\\tau_{\\mathrm{rel}}=0.12$ rejects isotropic "
                    "sensor noise ($\\rho\\approx 0.027$) without blocking 6-DoF gripper-and-object displacements ($\\rho\\ge 0.20$). "
                    "Because $u_k\\perp\\operatorname{span}(u_0,\\dots,u_{k-1})$ and $(u_k,g_k)$ never mutate once committed, "
                    "earlier coordinates suffer zero retroactive drift, high-contrast images ($\\lVert r\\rVert_2\\approx 4$) stay near "
                    "norm $s_f\\approx 2.1$, and millimetre displacements ($\\lVert r\\rVert_2\\approx 0.08$) gain $\\sim 6.25\\times$ SNR over $\\sigma=0.025$."
                ),
            },
            {
                "label": "Dual-Regime Semiparametric RLS, Spike-and-Slab, and Nadaraya-Watson Residual Prior",
                "latex": (
                    "\\begin{aligned}"
                    "\\omega_b(c)&=\\min(V_b,4)\\exp\\!\\left(-\\frac{\\lVert c-k_b^{\\mathrm{ctx}}\\rVert_2^2}{2h_{\\mathrm{nw}}^2}\\right),\\qquad "
                    "\\hat r_{\\mathrm{nw}}(c)=\\frac{\\sum_{b=0}^{B_t-1}\\omega_b(c)\\big(\\bar z_b-W_t\\bar k_b^{\\mathrm{ctx}}\\big)}"
                    "{\\kappa_{\\mathrm{nw}}+\\sum_{b=0}^{B_t-1}\\omega_b(c)},\\\\"
                    "\\hat z_{\\mathrm{base}}(c)&=\\hat z_{\\mathrm{ctx}}(c)+\\omega_{\\mathrm{nw}}(1-g_{\\mathrm{disc},t})\\hat r_{\\mathrm{nw}}(c),\\qquad "
                    "z_i^{\\mathrm{cond}}(c)=\\hat z_{\\mathrm{base}}(c)+\\gamma_t(c)\\big(\\hat\\mu_{q(i,b^*)}-\\hat z_{\\mathrm{ctx}}(k_{b^*}^{\\mathrm{ctx}})\\big)+\\epsilon_i."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Each context bucket maintains its exact unquantized latent mean $\\bar z_b\\in\\mathbb R^{d_z}$ alongside "
                    "discrete prototype counts. On continuous manifolds ($g_{\\mathrm{disc},t}\\to 0$), $(1-g_{\\mathrm{disc},t})\\hat r_{\\mathrm{nw}}(c)$ "
                    "interpolates piecewise-nonlinear contact, grasp, and friction residuals near visited states while shrinking "
                    "via $\\kappa_{\\mathrm{nw}}=0.06$ to global linear RLS far out-of-distribution; on discrete multimodal tasks "
                    "($g_{\\mathrm{disc},t}\\to 1$), exact empirical branch quantiles remain untouched."
                ),
            },
            {
                "label": "Fine-Salience Unnormalized Causal Reader",
                "latex": (
                    "\\begin{aligned}"
                    "s_t&=\\mathbb I\\!\\left[\\operatorname{rms}(\\bar x_t-\\operatorname{mean}(\\bar x_t)\\mathbf 1_{64})\\ge\\tau_{\\mathrm{con}}\\right],\\qquad "
                    "k_t=s_t\\,g_k W_k\\tilde x_t,\\qquad "
                    "f_t=k_t+\\tfrac12(k_t-k_{t-1})+(1-s_t)r_{t-1}."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Sparse 4-pixel-out-of-64 robotics action frames carry RMS contrast $\\frac14\\lVert 0.125\\,a\\rVert_2$, "
                    "which falls below $\\tau_{\\mathrm{con}}=0.03$ whenever $\\lVert a\\rVert_2<0.96$ during precision grasp and "
                    "hold phases. Lowering $\\tau_{\\mathrm{con}}$ from $0.03$ (`online-subspace-prior`) to $0.002$ "
                    "(`online-subspace-prior-fine-salience`) preserves fine control actions down to $\\lVert a\\rVert_2\\approx 0.064$ "
                    "while still gating exact zero-contrast uniform distractors on delayed-cue memory streams."
                ),
            },
            {
                "label": "Critic-Calmed Single-Arrival Adversarial and Markov-Bottlenecked Posterior Objectives",
                "latex": (
                    "\\begin{aligned}"
                    "\\mathcal L_D&=\\operatorname{softplus}(-D(x_t,c_{t-1}))+\\operatorname{softplus}(D(\\hat x_D,c_{t-1}))+"
                    "\\frac{\\lambda}{2}\\Big[(\\lVert g_r\\rVert_2-\\kappa)_+^2+(\\lVert g_f\\rVert_2-\\kappa)_+^2\\Big],\\\\"
                    "\\mathcal L_G&=\\operatorname{softplus}(-D(\\hat x_G,c_{t-1}))+\\lambda_{\\mathrm{rec}}\\,\\operatorname{BCE}(\\hat x_G,x_t)+"
                    "\\lambda_{\\mathrm{ctx}}\\,\\lVert W_{\\mathrm{ctx}}\\rVert_F^2."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Pairing critic step-size calming (`d_lr_mult=0.02`), stronger BCE posterior anchoring "
                    "($\\lambda_{\\mathrm{rec}}=100$, $\\lambda_{\\mathrm{ctx}}=5$), and causal Adam first-moment smoothing "
                    "($\\beta_1=0.4$, $\\eta=0.007$) prevents single-arrival critic gradients and shared convolutional bias "
                    "oscillations from perturbing previously acquired modes without replaying historical frames."
                ),
            },
        ],
        "pseudocode": [
            "Initialize fallback orthogonal encoder E_0, empty Gram-Schmidt subspace (k_0 = 0), float64 RLS (W_0 = 0, P_0 = I / lambda_rls), and empty prototype/context tables.",
            "For each chronological single-arrival step t (batch_size = 1, replay = none):",
            "  Before revealing x_t, serve y_hat = G(z_cond(i_serve, c_{t-1}), c_{t-1}) under an isolated eval RNG.",
            "  Reveal x_t; compute two-pass Gram-Schmidt residual r_t against U_{k_{t-1}} and stage (u_cand, g_cand) if ||r_t||_2 > tau_abs and ||r_t||_2 / (||v_t||_2 + 1e-3) > tau_rel.",
            "  Encode current target phi_t(x_t) using committed rows, any staged innovation row, and trailing JL rows E_0[K_sub:d_z].",
            "  Draw uniform row i_D; form z_D from RLS + spike-and-slab + (1 - g_disc) * r_nw(c_{t-1}) + gamma_t * quantile_residual + eps_{i_D}; update D at lr * 0.02 with BCAP cap.",
            "  Draw uniform row i_G; form posterior z_G = phi_t(x_t) + eps_{i_G}; update G with non-saturating GAN + 100 * BCE + 5 * ||W_ctx||_F^2 using Adam (beta_1 = 0.4).",
            "  After both optimizer steps succeed, commit any staged (u_cand, g_cand), update prototypes (<=32), context buckets and unquantized means z_bar_b (<=64), and rank-1 float64 RLS.",
            "  Advance detached causal state c_t = CausalStateMemory.write(x_t) without storing or replaying frames.",
        ],
        "training_details": {
            "loss": "Decoupled non-saturating logistic adversarial loss for $D$ and $G$, paired on $G$ with pixelwise BCE posterior reconstruction ($\\lambda_{\\mathrm{rec}}=100$) and Frobenius Markov-bottleneck regularization on direct generator context weights ($\\lambda_{\\mathrm{ctx}}=5$).",
            "optimizer": "Native Adam via `bcap_adam` with causal first-moment smoothing ($\\beta_1=0.4$, $\\beta_2=0.999$). Base `MoGParticlePrior` is frozen (`prior_update=\"frozen\"`); subspace, RLS, and Nadaraya-Watson buffers update analytically without optimizer state.",
            "learning_rates": "Generator base LR $0.007$ and calmed critic LR multiplier `d_lr_mult=0.02`, holding constant through `lr_anneal_start=0.55` before cosine-decaying to `lr_floor=0.06` (`network_lr_floor=0.06`).",
            "gradient_clipping": "None (`critic_grad_guard=False`). Critic Lipschitz control acts solely through the BCAP input-gradient cap.",
            "critic_penalty": "Fixed one-sided squared $L_2$ BCAP cap on real and subspace-prior fake inputs ($\\lambda=1.0$, $\\kappa=1.0$) every step; no critic EMA anchor.",
            "damping_and_guards": "No sparse-row latent damping (`latent_damping_max_rate=0.0`) and zero generator-center probes (`probe_law=\"no_generator_probes_v1\"`).",
            "online_memory_and_routing": "Detached `CausalStateMemory` ($d_c=16$, `normalize_key=False`, `key_gain=0.5`, `association_weight=0.0`, `contrast_threshold` $0.03$ or $0.002$) paired with `SubspaceResidualConditionalReconstruction` (`router=\"subspace_residual_conditional\"`, `subspace_dims=6`, `subspace_nu0=0.10`, `feature_scale=2.1`, `nw_weight=1.0`, `nw_bandwidth=0.65`, `nw_ridge=0.06`, `replay_policy=\"none\"`, `rehearsal_batch=0`).",
            "noise": "Zero scheduled critic-input or generator-output additive noise (`0.0`); intrinsic task MoG perturbation $\\sigma=0.025$ is preserved on every transformed particle row.",
            "averaging": "Live unaveraged generator weights (`ema_decay=0.0`, `serve_average=False`). Pre-arrival serving queries only committed subspace/RLS/NW sufficient statistics and preceding detached context $c_{t-1}$.",
        },
        "configuration_notes": [
            "`online-subspace-prior` branches from `online-residual-prior-linear-key`, keeping `CausalStateMemory` (`contrast_threshold=0.03`, `normalize_key=False`, `key_gain=0.5`) completely unchanged while introducing `SubspaceResidualConditionalReconstruction` and calmed single-arrival optimization (`d_lr_mult=0.02`, `betas=[0.4, 0.999]`, `lr=0.007`, `reconstruction_weight=100.0`).",
            "`online-subspace-prior-fine-salience` is a strict single-factor child of `online-subspace-prior` changing only `memory_policy.contrast_threshold` from `0.03` to `0.002` so sparse 4-of-64-pixel precision manipulation actions are not zeroed out near contact.",
            "When $d_c=0$ (`seq_minimal_dense4`, `seq_minimal_tokens4`, `seq_img_bars4`, `seq_img_dwell_bars4`, `seq_img_variable_dwell_bars4`, `seq_img_rare_mode_bars4`), `SubspaceResidualConditionalReconstruction` delegates sampling to `FeatureConditionalReconstruction` over the precision-whitened innovation basis, preserving exact marginal empirical quantiles.",
            "Staged innovation rows (`_pending_basis_row`, `_pending_basis_gain`) commit strictly after `GANTrainer.step` succeeds, guaranteeing exact transactional rollback if an optimizer update raises an exception.",
        ],
        "sources": [
            "particlestream/subspace_prior.py",
            "particlestream/residual_prior.py",
            "particlestream/feature_prior.py",
            "particlestream/memory.py",
            "particlestream/transport.py",
            "particlestream/tier1.py",
            "configs/forge/ideas/online-subspace-prior.json",
            "configs/forge/ideas/online-subspace-prior-fine-salience.json",
        ],
        "references": [
            {
                "title": "Numerics of Gram-Schmidt Orthogonalization",
                "url": "https://doi.org/10.1016/0024-3795(94)90493-6",
                "component": "Two-pass classical Gram-Schmidt reorthogonalization guaranteeing machine-precision orthonormality of online innovation basis rows.",
            },
            {
                "title": "On Estimating Regression",
                "url": "https://doi.org/10.1137/1109020",
                "component": "Nadaraya-Watson kernel regression estimator applied to unquantized per-bucket RLS residuals on continuous manifolds.",
            },
            {
                "title": "Smooth Regression Analysis",
                "url": "https://www.jstor.org/stable/25049340",
                "component": "Local kernel smoothing formulation paired with ridge shrinkage back to global linear RLS far from visited states.",
            },
            {
                "title": "Adjustment of an Inverse Matrix Corresponding to a Change in One Element of a Given Matrix",
                "url": "https://doi.org/10.1214/aoms/1177729893",
                "component": "Rank-1 Sherman-Morrison inverse-Gram update underlying the global parametric RLS baseline.",
            },
            {
                "title": "Adam: A Method for Stochastic Optimization",
                "url": "https://arxiv.org/abs/1412.6980",
                "component": "Causal first-moment smoothing (beta_1=0.4) stabilizing single-arrival convolutional decoder updates across mode switches.",
            },
        ],
    },
    "ResidualPrior-BCAP": {
        "overview": (
            "ResidualPrior-BCAP extends FeaturePrior-BCAP by replacing nearest-neighbor context lookup on "
            "unseen states with an online parametric residual conditional prior (`ResidualFeatureConditionalReconstruction`). "
            "A detached `float64` Sherman-Morrison Recursive Least Squares (RLS) affine predictor maintains global "
            "second-order sufficient statistics across all sequential arrivals without replay, paired with a "
            "Mahalanobis precision-weighted spike-and-slab prototype posterior that snaps along invariant subspaces "
            "on discrete compositional manifolds while staying linear on continuous physical manifolds. An "
            "empirical-Bayes kernel gate restores exact multimodal empirical quantiles at recurrent contexts, and a "
            "Markov bottleneck penalty prevents direct generator context-shortcut weights from bypassing the latent prior."
        ),
        "symbols": (
            "$G$ is the task-owned residual generator with input layer $W_{\\mathrm{fc}}=[W_z,\\,W_{\\mathrm{ctx}}]$, "
            "$D$ the raw-score convolutional critic, and $P$ the frozen $K$-row `MoGParticlePrior` ($K=32$, $d_z=8$, "
            "$\\sigma=0.025$) with base centers $\\mu_i^{\\mathrm{base}}\\in\\mathbb R^{d_z}$. At step $t$, "
            "$x_t\\in[0,1]^{1\\times 8\\times 8}$ is the single arriving target frame, $\\phi(x_t)=s_f W_{\\mathrm{enc}}(\\bar x_t-c_0\\mathbf 1_{64})\\in\\mathbb R^{d_z}$ "
            "its fixed orthogonal latent code ($s_f=2$, $c_0=0.5$), and $\\bar c_{t-1}=[\\tilde c_{t-1};\\,1]\\in\\mathbb R^{d_c+1}$ "
            "the clipped preceding causal context augmented with bias ($d_c=16$). $W_t\\in\\mathbb R^{d_z\\times(d_c+1)}$ and "
            "$P_t\\in\\mathbb R^{(d_c+1)\\times(d_c+1)}$ are the online RLS weights and inverse-Gram matrix ($\\lambda_{\\mathrm{rls}}=0.1$), "
            "$S_t\\in\\mathbb R^{d_z\\times d_z}$ and $q_t\\in\\mathbb R_{\\ge 0}$ are exponentially weighted ($\\alpha_S=0.05$) "
            "post-fit RLS residual covariance and nearest-prototype quantization variance ($\\varepsilon_S=10^{-4}$), "
            "$\\hat\\mu_m\\in\\mathbb R^{d_z}$ are online prototypes ($M_t\\le 32$), and $(k_b^{\\mathrm{ctx}},V_b,M_{b,m})$ "
            "are context keys, visit counts, and outcome masses ($C_{\\max}=64$, $\\kappa_{\\mathrm{visit}}=0.25$, $h_k=0.35$)."
        ),
        "equations": [
            {
                "label": "Causal Salient Key Projection (Normalized vs. Linear-Key Ablation)",
                "latex": (
                    "\\begin{aligned}"
                    "k_t^{\\mathrm{norm}}&=s_t\\frac{W_k\\tilde x_t}{\\lVert W_k\\tilde x_t\\rVert_2},\\qquad "
                    "k_t^{\\mathrm{lin}}=s_t\\,g_k W_k\\tilde x_t,\\qquad "
                    "f_t=k_t+\\tfrac12(k_t-k_{t-1})+(1-s_t)r_{t-1}."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "`online-residual-prior-bce` inherits unit-sphere normalization $k_t^{\\mathrm{norm}}$ (`normalize_key=True`), "
                    "whereas `online-residual-prior-linear-key` uses unnormalized salient projection $k_t^{\\mathrm{lin}}$ "
                    "(`normalize_key=False`, $g_k=0.5$) so disjoint pixel patches and continuous physical state-action pairs "
                    "preserve exact Euclidean superposition in context space while low-contrast arrivals ($s_t=0$) remain gated."
                ),
            },
            {
                "label": "Online Sherman-Morrison RLS Predictor and Residual Covariance",
                "latex": (
                    "\\begin{aligned}"
                    "g_t&=\\frac{P_{t-1}\\bar c_{t-1}}{1+\\bar c_{t-1}^\\top P_{t-1}\\bar c_{t-1}},\\qquad "
                    "W_t=W_{t-1}+\\big(\\phi(x_t)-W_{t-1}\\bar c_{t-1}\\big)g_t^\\top,\\qquad "
                    "P_t=P_{t-1}-g_t\\bar c_{t-1}^\\top P_{t-1},\\\\"
                    "e_t^{\\mathrm{post}}&=\\phi(x_t)-W_t\\bar c_{t-1},\\qquad "
                    "S_t=(1-\\alpha_S)S_{t-1}+\\alpha_S\\,e_t^{\\mathrm{post}}(e_t^{\\mathrm{post}})^\\top,\\qquad "
                    "q_t=(1-\\alpha_S)q_{t-1}+\\alpha_S\\min_m\\lVert\\phi(x_t)-\\hat\\mu_m\\rVert_2^2."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Rank-1 Sherman-Morrison updates accumulate exact ridge-regularized linear sufficient statistics across "
                    "correlated `batch_size=1` streams without forgetting earlier episodes. After warm-up ($t>d_c$), $S_t$ "
                    "captures directional post-fit residual covariance while $q_t$ tracks continuous vs. discrete target dispersion."
                ),
            },
            {
                "label": "Mahalanobis Spike-and-Slab Prototype Posterior and Kernel-Gated Residual Quantiles",
                "latex": (
                    "\\begin{aligned}"
                    "\\hat z_{\\mathrm{lin}}(c)&=W_t\\bar c,\\qquad "
                    "\\tilde S_t=\\tfrac12(S_t+S_t^\\top)+\\varepsilon_S I,\\qquad "
                    "m^*(c)=\\arg\\min_{0\\le m<M_t}(\\hat\\mu_m-\\hat z_{\\mathrm{lin}}(c))^\\top\\tilde S_t^{-1}(\\hat\\mu_m-\\hat z_{\\mathrm{lin}}(c)),\\\\"
                    "\\beta_t(c)&=\\exp\\!\\left(-\\left(\\frac{q_t}{\\rho_v\\operatorname{tr}(S_t)+10^{-5}}\\right)^2\\right)"
                    "\\sigma\\!\\left(\\tfrac12\\log\\frac{\\det(\\tilde S_t+\\sigma_{\\mathrm{slab}}^2 I)}{\\det(\\tilde S_t)}-\\tfrac12 d_{\\tilde S_t}^2(\\hat\\mu_{m^*(c)},\\hat z_{\\mathrm{lin}}(c))\\right),\\\\"
                    "\\hat z_{\\mathrm{ctx}}(c)&=(1-\\beta_t(c))\\hat z_{\\mathrm{lin}}(c)+\\beta_t(c)\\hat\\mu_{m^*(c)},\\qquad "
                    "z_i^{\\mathrm{cond}}(c)=\\hat z_{\\mathrm{ctx}}(c)+\\gamma_t(c)\\big(\\hat\\mu_{q(i,b^*)}-\\hat z_{\\mathrm{ctx}}(k_{b^*}^{\\mathrm{ctx}})\\big)+\\epsilon_i."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Along exact linear invariants where $\\tilde S_t$ has near-zero variance, $\\tilde S_t^{-1}$ enforces "
                    "the invariant while down-weighting nonlinear carry/parity directions ($\\rho_v=0.15$, $\\sigma_{\\mathrm{slab}}=1.0$). "
                    "Using the shared discreteness gate $g_{\\mathrm{disc},t}=\\exp(-(q_t/(\\rho_v\\operatorname{tr}(S_t)+10^{-5}))^2)$, "
                    "continuous manifolds ($q_t/\\operatorname{tr}(S_t)\\gg\\rho_v$) satisfy $\\beta_t(c)\\to 0$ and $\\gamma_t(c)\\to 0$ "
                    "for smooth extrapolation, while recurrent multimodal contexts ($V_{b^*}>1$) recover exact branch quantiles via "
                    "$\\gamma_t(c)=\\frac{V_{b^*}-1}{V_{b^*}-1+\\kappa_{\\mathrm{visit}}}g_{\\mathrm{disc},t}\\exp(-\\lVert c-k_{b^*}^{\\mathrm{ctx}}\\rVert_2^2/h_k^2)$."
                ),
            },
            {
                "label": "Markov-Bottlenecked Generator and Critic Objectives",
                "latex": (
                    "\\begin{aligned}"
                    "\\mathcal L_D&=\\operatorname{softplus}(-D(x_t,c_{t-1}))+\\operatorname{softplus}(D(\\hat x_D,c_{t-1}))+"
                    "\\frac{\\lambda}{2}\\Big[(\\lVert g_r\\rVert_2-\\kappa)_+^2+(\\lVert g_f\\rVert_2-\\kappa)_+^2\\Big],\\\\"
                    "\\mathcal L_G&=\\operatorname{softplus}(-D(\\hat x_G,c_{t-1}))+\\lambda_{\\mathrm{rec}}\\,\\operatorname{BCE}(\\hat x_G,x_t)+"
                    "\\lambda_{\\mathrm{ctx}}\\,\\lVert W_{\\mathrm{ctx}}\\rVert_F^2."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Because $G$ trains on current-observation posterior codes $z_{i_G}^{\\mathrm{post}}=\\phi(x_t)+\\epsilon_{i_G}$, "
                    "penalizing the direct context-input block $W_{\\mathrm{ctx}}$ ($\\lambda_{\\mathrm{ctx}}=5$, $\\lambda_{\\mathrm{rec}}=10$) "
                    "prevents $G$ from memorizing seen contexts around $z$ and ensures held-out states decode cleanly through $z_i^{\\mathrm{cond}}(c)$."
                ),
            },
        ],
        "pseudocode": [
            "Initialize fixed orthogonal encoder W_enc, float64 RLS (W_0 = 0, P_0 = I / lambda_rls), S_0 = 1e-2 I, q_0 = 0, and empty prototype/context tables.",
            "For each chronological single-arrival step t (batch_size = 1, replay = none):",
            "  Before revealing x_t, serve y_hat = G(z_cond(i_serve, c_{t-1}), c_{t-1}) under an isolated eval RNG.",
            "  Reveal single target frame x_t and encode phi(x_t) = s_f W_enc (x_bar_t - c_0).",
            "  Draw uniform row i_D; compute z_ctx(c_{t-1}) via RLS + Mahalanobis spike-and-slab and add kernel-gated live residual quantile to form z_D.",
            "  Update critic D with non-saturating logistic loss on x_t and stop_gradient(G(z_D, c_{t-1})) plus BCAP input-gradient penalty.",
            "  Draw uniform row i_G; form posterior code z_G = phi(x_t) + eps_{i_G}; update G with adversarial loss + 10 * BCE + 5 * ||W_ctx||_F^2.",
            "  After both updates succeed, commit phi(x_t) to prototype table (<=32) and c_{t-1} to context table (<=64), then run rank-1 Sherman-Morrison RLS and EMA updates for (W_t, P_t, S_t, q_t).",
            "  Advance detached causal state c_t = CausalStateMemory.write(x_t) without storing or replaying frames.",
        ],
        "training_details": {
            "loss": "Decoupled non-saturating logistic adversarial loss for $D$ and $G$, paired on $G$ with pixelwise BCE posterior reconstruction ($\\lambda_{\\mathrm{rec}}=10$) and Frobenius regularization on $G$'s direct context-input weights ($\\lambda_{\\mathrm{ctx}}=5$).",
            "optimizer": "Native Adam via `bcap_adam` ($\\beta_1=0$, $\\beta_2=0.99$, `prior_betas=(0, 0.99)`). Base `MoGParticlePrior` is frozen (`prior_update=\"frozen\"`); RLS and covariance buffers update analytically in `float64` without optimizer state.",
            "learning_rates": "Base LR $0.005$ with `critic_lr_mult=1.0`, `prior_lr_mult=1.0`, holding constant through `lr_anneal_start=0.5` before cosine-decaying to `lr_floor=0.05`.",
            "gradient_clipping": "None (`critic_grad_guard=False`). Critic Lipschitz control acts solely through the BCAP input-gradient cap.",
            "critic_penalty": "Fixed one-sided squared $L_2$ BCAP cap on real and residual-prior fake inputs ($\\lambda=1.0$, $\\kappa=1.0$) every step; no critic EMA anchor.",
            "damping_and_guards": "No sparse-row latent damping (`latent_damping_max_rate=0.0`) and zero generator-center probes (`probe_law=\"no_generator_probes_v1\"`).",
            "online_memory_and_routing": "Detached `CausalStateMemory` ($d_c=16$, `association_weight=0.0`) plus `ResidualFeatureConditionalReconstruction` (`rls_ridge=0.1`, `covariance_ema=0.05`, `slab_scale=1.0`, `variance_ratio_scale=0.15`, `visit_shrinkage=0.25`, `kernel_bandwidth=0.35`, `replay_policy=\"none\"`, `rehearsal_batch=0`).",
            "noise": "Zero scheduled critic-input or generator-output additive noise (`0.0`); intrinsic task MoG perturbation $\\sigma=0.025$ is preserved on every transformed particle row.",
            "averaging": "Live unaveraged generator weights (`ema_decay=0.0`, `serve_average=False`). Pre-arrival serving queries only committed sufficient statistics and preceding detached context $c_{t-1}$.",
        },
        "configuration_notes": [
            "`online-residual-prior-bce` isolates the prior/transport change relative to `online-feature-prior-bce`, keeping `CausalStateMemory` (`normalize_key=True`) and every optimizer/loss override unchanged while replacing `FeatureConditionalReconstruction` with `ResidualFeatureConditionalReconstruction`.",
            "`online-residual-prior-linear-key` is a single-factor child of `online-residual-prior-bce` setting `normalize_key=False` (`key_gain=0.5`) in `CausalStateMemory`, preserving linear Euclidean norm across multi-patch compositional frames and continuous state-action encodings.",
            "When $d_c=0$ (`seq_minimal_dense4`, `seq_minimal_tokens4`, `seq_img_bars4`, `seq_img_dwell_bars4`, `seq_img_variable_dwell`, `seq_img_rare_mode`), `ResidualFeatureConditionalReconstruction` delegates directly to `FeatureConditionalReconstruction` and preserves exact marginal empirical quantiles.",
            "Zero trainable `nn.Parameter` tensors or random weights are added by the residual prior; all updates are deterministic closed-form matrix operations committed strictly after `GANTrainer.step` succeeds.",
        ],
        "sources": [
            "particlestream/residual_prior.py",
            "particlestream/feature_prior.py",
            "particlestream/memory.py",
            "particlestream/transport.py",
            "particlestream/tier1.py",
            "configs/forge/ideas/online-residual-prior-bce.json",
            "configs/forge/ideas/online-residual-prior-linear-key.json",
        ],
        "references": [
            {
                "title": "Adjustment of an Inverse Matrix Corresponding to a Change in One Element of a Given Matrix",
                "url": "https://doi.org/10.1214/aoms/1177729893",
                "component": "Rank-1 Sherman-Morrison inverse-Gram recursion powering the exact online batch_size=1 RLS predictor.",
            },
            {
                "title": "Bayesian Variable Selection in Linear Regression",
                "url": "https://doi.org/10.1080/01621459.1988.10478693",
                "component": "Spike-and-slab Bayes factor formulation adapted for precision-weighted discrete prototype vs. continuous slab gating.",
            },
            {
                "title": "Deep Variational Information Bottleneck",
                "url": "https://arxiv.org/abs/1612.00410",
                "component": "Information-bottleneck motivation for regularizing direct generator context-shortcut weights so predictions route through the latent code.",
            },
            {
                "title": "Generative Adversarial Networks",
                "url": "https://arxiv.org/abs/1406.2661",
                "component": "Non-saturating logistic adversarial objective paired with the residual conditional prior.",
            },
        ],
    },
    "FeaturePrior-BCAP": {
        "overview": (
            "FeaturePrior-BCAP pairs the fixed BCAP critic input-gradient cap and a detached causal state "
            "reader with a bounded observed-feature conditional prior. A fixed deterministic orthogonal "
            "8×64 encoder projects each arriving 1×8×8 frame into latent space, updating at most 32 online "
            "feature prototypes and 64 context-outcome histograms only after successful optimizer steps. "
            "The generator trains its unchanged decoder via a weight-10 BCE current-observation posterior "
            "while freezing the storage prior; critic updates and causal pre-arrival serving draw empirical "
            "feature quantiles at the nearest observed past context without generator probes or frame replay."
        ),
        "symbols": (
            "$G$ is the task-owned residual generator, $D$ the raw-score convolutional critic, and $P$ the "
            "frozen $K$-row `MoGParticlePrior` ($K=32$, $d_z=8$, intrinsic scale $\\sigma=0.025$) with base "
            "centers $\\mu_i^{\\mathrm{base}}\\in\\mathbb R^{d_z}$. At chronological step $t$, $x_t\\in[0,1]^{1\\times 8\\times 8}$ "
            "is the single arriving frame, $\\bar x_t=\\operatorname{vec}(x_t)\\in\\mathbb R^{64}$, "
            "$\\tilde x_t=\\bar x_t-\\operatorname{mean}(\\bar x_t)\\mathbf 1_{64}$ is the DC-centered vector, "
            "and $c_{t-1}\\in\\mathbb R^{d_c}$ ($d_c=16$) is the detached causal context from preceding arrivals. "
            "$W_k,W_v\\in\\mathbb R^{d_c\\times 64}$ are fixed detached reader projections, $s_t\\in\\{0,1\\}$ "
            "is the contrast salience gate ($\\tau_{\\mathrm{con}}=0.03$), $r_t$ is the retained salient key, "
            "$a_t\\in\\{0,\\dots,H_{\\mathrm{age}}\\}$ ($H_{\\mathrm{age}}=8$) is the repetition-age counter, and "
            "$\\omega_{\\mathrm{assoc}}$ is the associative read weight ($0$ in `online-feature-prior-bce`). "
            "$W_{\\mathrm{enc}}\\in\\mathbb R^{d_z\\times 64}$ is a fixed row-normalized deterministic orthogonal "
            "projection ($s_f=2$, $c_0=0.5$), $\\hat\\mu_m\\in\\mathbb R^{d_z}$ are online feature prototypes "
            "($\\tau_f=0.15$), $k_b^{\\mathrm{ctx}}\\in\\mathbb R^{d_c}$ and $N_{b,m}\\in\\mathbb N_0$ are "
            "context keys and outcome counts ($C_{\\max}=64$, $\\tau_c=0.25$), $(a)_+=\\max(a,0)$, and "
            "$\\operatorname{softplus}(a)=\\log(1+e^a)$."
        ),
        "equations": [
            {
                "label": "Detached Causal State Reader",
                "latex": (
                    "\\begin{aligned}"
                    "k_t&=s_t\\frac{W_k\\tilde x_t}{\\lVert W_k\\tilde x_t\\rVert_2},\\qquad "
                    "r_t=s_t k_t+(1-s_t)r_{t-1},\\qquad "
                    "f_t=k_t+\\tfrac12(k_t-k_{t-1})+(1-s_t)r_{t-1}+\\omega_{\\mathrm{assoc}}\\tanh(M_t q_t),\\\\"
                    "c_{t,0:d_c-1}&=s_{\\mathrm{ctx}}\\operatorname{clip}(f_{t,0:d_c-1},-1,1),\\qquad "
                    "c_{t,d_c-1}=s_{\\mathrm{ctx}}\\frac{a_t}{H_{\\mathrm{age}}}."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Low-contrast arrivals ($s_t=0$) preserve the most recent salient key $r_{t-1}$ instead "
                    "of overwriting context with normalized sensor noise, while the final coordinate encodes "
                    "bounded repetition age $a_t/H_{\\mathrm{age}}$ ($s_{\\mathrm{ctx}}=3$). All writes run "
                    "detached from optimizer autograd."
                ),
            },
            {
                "label": "Observed-Feature Encoding and Empirical Quantile Prior",
                "latex": (
                    "\\begin{aligned}"
                    "\\phi(x_t)&=s_f\\,W_{\\mathrm{enc}}(\\bar x_t-c_0\\mathbf 1_{64}),\\qquad "
                    "b^*=\\arg\\min_{0\\le b<C_{\\mathrm{occ}}}\\lVert k_b^{\\mathrm{ctx}}-\\tilde c_{t-1}\\rVert_2,\\\\"
                    "m^*(i)&=\\min\\!\\left(K-1,\\;\\inf\\left\\{m\\ge 0:\\frac{\\sum_{j=0}^m N_{b^*,j}}{\\sum_{j=0}^{K-1}N_{b^*,j}}>\\frac{i+0.5}{K}\\right\\}\\right),\\\\"
                    "z_i^{\\mathrm{cond}}&=\\hat\\mu_{m^*(i)}+(z_i^{\\mathrm{raw}}-\\mu_i^{\\mathrm{base}}),\\qquad "
                    "z_i^{\\mathrm{post}}=\\phi(x_t)+(z_i^{\\mathrm{raw}}-\\mu_i^{\\mathrm{base}})."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Every draw retains its exact task-owned Gaussian perturbation $\\epsilon_i=z_i^{\\mathrm{raw}}-\\mu_i^{\\mathrm{base}}$. "
                    "Critic sampling and live serving map particle row $i$ through the midpoint empirical CDF at "
                    "the nearest observed context $b^*$ (falling back to $z_i^{\\mathrm{raw}}$ before observations "
                    "exist), whereas the generator training step substitutes the current-observation code $\\phi(x_t)$."
                ),
            },
            {
                "label": "Single-Arrival Adversarial and Posterior Reconstruction Objectives",
                "latex": (
                    "\\begin{aligned}"
                    "\\mathcal L_D&=\\operatorname{softplus}(-D(x_t,c_{t-1}))+\\operatorname{softplus}(D(\\hat x_D,c_{t-1}))+"
                    "\\frac{\\lambda}{2}\\Big[(\\lVert g_r\\rVert_2-\\kappa)_+^2+(\\lVert g_f\\rVert_2-\\kappa)_+^2\\Big],\\\\"
                    "\\mathcal L_G&=\\operatorname{softplus}(-D(\\hat x_G,c_{t-1}))+\\lambda_{\\mathrm{rec}}\\,\\operatorname{BCE}(\\hat x_G,x_t)."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Here $\\hat x_D=G(z_{i_D}^{\\mathrm{cond}},c_{t-1})$ is detached for the critic step and "
                    "$\\hat x_G=G(z_{i_G}^{\\mathrm{post}},c_{t-1})$ trains the image decoder at $\\lambda_{\\mathrm{rec}}=10$, "
                    "$\\lambda=1$, and $\\kappa=1$ with uniform single-particle draws $i_D,i_G\\sim\\mathcal U\\{0,\\dots,K-1\\}$."
                ),
            },
        ],
        "pseudocode": [
            "Initialize fixed orthogonal encoder W_enc, empty prototype table (<=32) and context table (<=64).",
            "For each chronological single-arrival step t:",
            "  Before revealing x_t, serve y_hat = G(z_cond(i_serve, c_{t-1}), c_{t-1}) under an isolated eval RNG.",
            "  Reveal single target frame x_t (batch_size = 1) and encode latent feature phi(x_t) = s_f W_enc (x_bar_t - c_0).",
            "  Draw uniform row i_D and transform z_D = z_cond(i_D, c_{t-1}) via empirical quantiles at nearest context b*.",
            "  Evaluate non-saturating critic loss on x_t and stop_gradient(G(z_D, c_{t-1})) plus fixed BCAP penalty; step D.",
            "  With D fixed, draw uniform row i_G and form current-observation posterior code z_G = phi(x_t) + eps_{i_G}.",
            "  Evaluate non-saturating generator loss + 10 * BCE(G(z_G, c_{t-1}), x_t); step G while keeping base prior P frozen.",
            "  After both updates succeed, merge phi(x_t) into nearest prototype mu_hat_m (tol 0.15) and c_{t-1} into context bucket b (tol 0.25, LRU cap 64); increment N_{b, m}.",
            "  Advance detached causal state c_t = CausalStateMemory.write(x_t) without storing or replaying frames.",
        ],
        "training_details": {
            "loss": "Decoupled non-saturating logistic adversarial loss for $D$ and $G$, paired on $G$ with pixelwise binary cross-entropy reconstruction ($\\lambda_{\\mathrm{rec}}=10$) from the current-observation feature posterior.",
            "optimizer": "Native Adam via `bcap_adam` ($\\beta_1=0$, $\\beta_2=0.99$, `prior_betas=(0, 0.99)`). The base MoG storage table is frozen (`prior_update=\"frozen\"`).",
            "learning_rates": "Base LR $0.005$ with `critic_lr_mult=1.0`, `prior_lr_mult=1.0`, holding constant through $50\\%$ of the task horizon (`lr_anneal_start=0.5`) before cosine-decaying to $5\\%$ (`lr_floor=0.05`).",
            "gradient_clipping": "None (`critic_grad_guard=False`). Critic Lipschitz control acts smoothly through the BCAP input-gradient loss penalty.",
            "critic_penalty": "Fixed one-sided squared $L_2$ BCAP cap on real and conditional-prior fake inputs ($\\lambda=1.0$, $\\kappa=1.0$) every update; no EMA critic anchor.",
            "damping_and_guards": "No A2 sparse-row damping (`latent_damping_max_rate=0.0`) and no generator-center probes (`probe_law=\"no_generator_probes_v1\"`).",
            "online_memory_and_routing": "Detached `CausalStateMemory` ($d_c=16$, `association_weight=0.0`, `contrast_threshold=0.03`, `change_threshold=0.04`, `age_horizon=8`) plus `FeatureConditionalReconstruction` (up to $32$ feature means and $64$ LRU context-outcome histograms; `replay_policy=\"none\"`, `rehearsal_batch=0`).",
            "noise": "Zero scheduled critic-input or generator-output additive noise (`noise_std_start=0.0`, `gen_output_noise_std=0.0`). Intrinsic task MoG perturbation $\\sigma=0.025$ is preserved on every transformed row.",
            "averaging": "Live unaveraged generator weights (`ema_decay=0.0`, `serve_average=False`). Serving uses only preceding context $c_{t-1}$ and previously committed feature/outcome summaries.",
        },
        "configuration_notes": [
            "FeaturePrior-BCAP established the first Tier 1-qualifying global recipe (`online-feature-prior-bce`: 10/10 required passes, 1/10 stretch passes on `single_stream_stability` view 10), serving as the direct parent of both ResidualPrior-BCAP and GAIL-BCAP.",
            "Unlike competitive or context-balanced transport, FeaturePrior-BCAP executes zero extra generator forward probes per update: both $D$ and $G$ draw one uniform particle row at `batch_size=1`.",
            "Feature prototypes and context-outcome counts commit strictly after the single-arrival critic and generator updates complete, preserving causal pre-arrival serving.",
            "Because unseen contexts fall back to 1-NN lookup over at most 64 observed context keys while $G$ trains on $\\phi(x_t)+\\epsilon$, compositional held-out extrapolation (`addition2bit`, `kernel2x2`, `velocity_extrapolation`) and high-DoF continuous state-action manifolds (`robotics`) remain open stretch frontiers.",
        ],
        "sources": [
            "particlestream/feature_prior.py",
            "particlestream/memory.py",
            "particlestream/transport.py",
            "particlestream/tier1.py",
            "configs/forge/ideas/online-feature-prior-bce.json",
        ],
        "references": [
            {
                "title": "Generative Adversarial Networks",
                "url": "https://arxiv.org/abs/1406.2661",
                "component": "Non-saturating logistic adversarial objective paired with the conditional feature posterior.",
            },
            {
                "title": "Adam: A Method for Stochastic Optimization",
                "url": "https://arxiv.org/abs/1412.6980",
                "component": "First-order moment optimizer used for generator and critic updates.",
            },
            {
                "title": "On the regularization of Wasserstein GANs",
                "url": "https://arxiv.org/abs/1709.08894",
                "component": "Related one-sided squared hinge gradient penalty; BCAP evaluates the cap directly at real and generated endpoints.",
            },
            {
                "title": "Using Fast Weights to Attend to the Recent Past",
                "url": "https://arxiv.org/abs/1610.06258",
                "component": "Inspiration for bounded detached fast-weight state maintained alongside slow network weights.",
            },
        ],
    },
    "GAIL-BCAP": {
        "overview": (
            "GAIL-BCAP translates stabilized generative adversarial imitation control into ParticleStream's "
            "strict `batch_size=1` online world-modeling contract. It pairs a Hinge margin critic and fixed BCAP "
            "input-gradient cap with a prioritized survival-plus-style generator objective: unbounded linear Hinge "
            "generator slopes are canceled in favor of pixelwise BCE reconstruction minus a bounded sigmoid critic "
            "style bonus modulated by causal return advantage and active-prototype policy entropy. Frontier recipes "
            "upgrade conditional transport with precision-whitened two-pass Gram-Schmidt innovation subspaces, "
            "pre-birth debiased `float64` Sherman-Morrison RLS, Mahalanobis spike-and-slab prototype gating with "
            "consistent discrete-fork branch anchoring, unbucketed 512-point Matérn-3/2 kernel ridge regression "
            "(`KRR`), orthogonal previous-frame value readout and causal $\\alpha$-$\\beta$ barycentric kinematic "
            "tracking in `CausalStateMemory`, causal Empirical-Bayes Dirichlet-precision gated Fisher-Sobolev $H^1$ "
            "and quartic boundary calibration, 2D lattice-harmonic latent lifting with $\\mathbb Z^2$ shift-equivariant "
            "moment regularization on localized wavepackets, and polynomial Primal-Dual Schedule-Free generator iterate averaging."
        ),
        "symbols": (
            "$G$ is the task-owned residual generator, $D$ the raw-score convolutional critic, and $P$ the frozen "
            "$K$-row `MoGParticlePrior` ($K=32$, $d_z=8$, intrinsic scale $\\sigma=0.025$). At chronological arrival $t$, "
            "$x_t\\in[0,1]^{1\\times 8\\times 8}$ is the single target frame, $c_{t-1}\\in\\mathbb R^{d_c}$ ($d_c=16$) is "
            "the detached causal-state context (`write_lr` $\\eta_w\\in\\{0.5,0.35\\}$, contrast gate $\\tau_{\\mathrm{con}}\\in\\{0.03,0.005,0.002\\}$, "
            "optional previous-frame value weight $w_{\\mathrm{pv}}\\in\\{0,1\\}$, optional $\\alpha$-$\\beta$ barycentric kinematic gains "
            "$(\\alpha_{\\mathrm{kf}},\\beta_{\\mathrm{kf}})=(0.20,0.04)$), "
            "and $\\phi_t(x_t)\\in\\mathbb R^{d_z}$ encodes $v_t=\\bar x_t-c_0\\mathbf 1_{64}$ ($c_0=0.5$) via fixed "
            "orthogonal rows ($K_{\\mathrm{sub}}=0$, $s_f\\in\\{2,4\\}$), $K_{\\mathrm{sub}}=6$ precision-whitened "
            "Gram-Schmidt innovation rows ($\\nu_0=0.10$, $s_f=2.1$) plus 2 deterministic JL rows, or 2D lattice-harmonic "
            "barycentric coordinates ($s_{\\mathrm{wp}}=2.5$) on localized mass-conserving wavepackets ($\\alpha_{\\mathrm{wp},t}=1$). "
            "$\\lambda_{\\mathrm{rec}}\\in\\{10,100\\}$ is the survival BCE weight, $w_{\\mathrm{style}}\\in\\{0.1,0.05,0.002\\}$ "
            "the bounded style bonus weight, $\\sigma_{\\mathrm{sig}}(u)=(1+e^{-u})^{-1}$ the logistic sigmoid, "
            "$\\gamma_R=0.99$ the causal return discount, $R_t$ and $(b_t,v_t)$ the post-commit return trace and moments, "
            "$\\alpha_A\\in\\{0.25,0\\}$ the advantage scale, $\\lambda_{\\mathrm{ctx}}\\in\\{0,5\\}$ the Markov bottleneck weight, "
            "$\\eta_H=0.01$ the active-prototype policy entropy bonus, $(M_t,P_{t,r},\\Sigma_t)$ the per-coordinate `float64` RLS and residual "
            "covariance state, $(K_{\\mathrm{mat}},h_{\\mathrm{res}},\\lambda_{\\mathrm{krr}},C_{\\mathrm{krr}})$ the Matérn-3/2 KRR kernel "
            "($h_{\\mathrm{res}}\\in\\{1.15,0.72\\}$, $\\lambda_{\\mathrm{krr}}\\in\\{0.06,0.025\\}$, $C_{\\mathrm{krr}}\\in\\{0,512\\}$), "
            "$\\alpha_{\\mathrm{dir},t}\\in(0,1]$ the causal Dirichlet-precision constant-field gate ($\\sigma_\\nabla=10^{-7}$, "
            "$w_{\\mathrm{sob}}=36$, $\\gamma_{\\mathrm{bnd}}=0.85$, $\\lambda_{\\mathrm{dir,ctx}}=55$, $\\Delta c_{\\mathrm{dir}}=0.02$), "
            "$(w_{\\mathrm{eq}},w_{\\mathrm{mom}},w_{\\mathrm{mse}},\\lambda_{\\mathrm{wp,ctx}},N_{\\mathrm{sh}},\\Delta c_{\\mathrm{wp}})=(3.0,0.07,2.5,80,6,0.205)$ "
            "the wavepacket shift-equivariant calibration parameters, "
            "and $(z_t^{\\mathrm{sf}},x_t^{\\mathrm{sf}},y_t^{\\mathrm{sf}})$ the Schedule-Free dual, primal, and interpolation generator states."
        ),
        "equations": [
            {
                "label": "Hinge-BCAP Critic, Prioritized Survival + Bounded Style Bonus, and Markov Bottleneck",
                "latex": (
                    "\\begin{aligned}"
                    "\\mathcal L_D&=\\big(1-D(x_t,c_{t-1})\\big)_++\\big(1+D(\\hat x_D,c_{t-1})\\big)_++"
                    "\\frac{\\lambda}{2}\\Big[(\\lVert g_r\\rVert_2-\\kappa)_+^2+(\\lVert g_f\\rVert_2-\\kappa)_+^2\\Big],\\\\"
                    "g_{t-1}&=1+\\alpha_A\\tanh\\!\\left(\\frac{R_{t-1}-b_{t-1}}{\\sqrt{v_{t-1}+10^{-10}}}\\right),\\\\"
                    "\\mathcal L_G&=\\lambda_{\\mathrm{rec}}\\Big[\\operatorname{BCE}(\\hat x_G,x_t)-"
                    "w_{\\mathrm{style}}\\,g_{t-1}\\,\\sigma_{\\mathrm{sig}}\\big(D(\\hat x_G,c_{t-1})\\big)\\Big]+"
                    "\\lambda_{\\mathrm{ctx}}\\lVert W_{fc,\\mathrm{ctx}}\\rVert_F^2."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Canceling the default unbounded Hinge generator term $-D(\\hat x_G,c_{t-1})$ prevents single-sample "
                    "critic spikes from destabilizing the shared decoder; scaling $\\lambda_{\\mathrm{rec}}=100$ with "
                    "$w_{\\mathrm{style}}=0.002$ (`online-gail-bcap-subspace` onward) strengthens "
                    "posterior pixel anchoring $10\\times$ while keeping effective style weight $\\lambda_{\\mathrm{rec}}w_{\\mathrm{style}}=0.20$ calibrated."
                ),
            },
            {
                "label": "Entropy-Regularized Active-Prototype Quantile Policy",
                "latex": (
                    "\\begin{aligned}"
                    "\\mathcal A_{b^*}&=\\big\\{m\\in\\{0,\\dots,M_{\\mathrm{occ}}-1\\}:N_{b^*,m}>0\\big\\},\\\\"
                    "\\tilde p_{b^*,m}&=(1-\\eta_H)\\frac{N_{b^*,m}}{\\sum_{j\\in\\mathcal A_{b^*}}N_{b^*,j}}+"
                    "\\eta_H\\frac{\\mathbb I(m\\in\\mathcal A_{b^*})}{|\\mathcal A_{b^*}|},\\\\"
                    "m^*(i)&=\\min\\!\\left(m_{\\max}(\\mathcal A_{b^*}),\\;\\inf\\left\\{m\\ge 0:"
                    "\\sum_{j=0}^m \\tilde p_{b^*,j}>\\frac{i+0.5}{K}\\right\\}\\right)."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Mixing $\\eta_H=0.01$ uniform probability strictly over active prototypes $\\mathcal A_{b^*}$ at "
                    "the nearest context bucket $b^*$ prevents finite-sample lock-out on rare discrete branches (`5:5:5:1`) "
                    "and stabilizes recurrent state-action buckets in multi-episode robotics."
                ),
            },
            {
                "label": "Precision-Whitened Innovation Subspace, Spike-and-Slab RLS, and Matérn-3/2 KRR Transport",
                "latex": (
                    "\\begin{aligned}"
                    "r_t&=\\big(I-U_{k_{t-1}}U_{k_{t-1}}^\\top\\big)^2 v_t,\\qquad "
                    "g_{k_{t-1}}=\\big(\\lVert r_t\\rVert_2^2+\\nu_0^2\\big)^{-1/2},\\qquad "
                    "\\kappa_{3/2}(d)=\\left(1+\\frac{\\sqrt{3}\\,d}{h_{\\mathrm{res}}}\\right)\\exp\\!\\left(-\\frac{\\sqrt{3}\\,d}{h_{\\mathrm{res}}}\\right),\\\\"
                    "\\hat r_{\\mathrm{krr}}(c)&=k_{3/2}(c)^\\top\\!\\left(K_{3/2}+\\operatorname{diag}\\!\\left(\\frac{\\lambda_{\\mathrm{krr}}}{\\min(V_b,4)}\\right)\\right)^{-1}R_{\\mathrm{rls}},\\qquad "
                    "\\hat z_{\\mathrm{base}}(c)=\\hat z_{\\mathrm{ctx}}(c)+(1-g_{\\mathrm{disc},t})\\hat r_{\\mathrm{krr}}(c),\\\\"
                    "z_i^{\\mathrm{cond}}(c)&=\\hat z_{\\mathrm{base}}(c)+\\gamma_t(c)\\Big(\\hat\\mu_{m^*(i)}-\\hat z_{\\mathrm{ctx}}(k_{b^*}^{\\mathrm{ctx}})\\Big)+\\epsilon_i."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "In `online-gail-bcap-subspace` and `online-gail-bcap-schedule-free`, $K_{\\mathrm{sub}}=6$ anchored "
                    "Gram-Schmidt directions capture exact active target subspaces while solving an exact $B_t\\times B_t$ "
                    "(`B_t <= 64`) `float64` Matérn-3/2 kernel ridge system over unquantized per-bucket RLS residuals "
                    "$R_{\\mathrm{rls},b}=\\bar z_b-M_t\\bar k_b^{\\mathrm{ctx}}$ eliminates Nadaraya-Watson cluster-density bias near contact."
                ),
            },
            {
                "label": "Horizon-Invariant Causal Exponential Cooling and Primal-Dual Schedule-Free Generator",
                "latex": (
                    "\\begin{aligned}"
                    "\\eta(t)&=\\eta_0\\,r_{\\mathrm{dec}}^{t/S_{\\mathrm{dec}}},\\qquad "
                    "c_t=\\min\\!\\left(1,\\,\\max\\!\\left(\\frac{k_{\\mathrm{sf}}}{t+k_{\\mathrm{sf}}-1},\\,c_{\\min}\\right)\\right),\\\\"
                    "y_t^{\\mathrm{sf}}&=(1-\\beta_{\\mathrm{sf}})z_{t-1}^{\\mathrm{sf}}+\\beta_{\\mathrm{sf}}x_{t-1}^{\\mathrm{sf}},\\qquad "
                    "z_t^{\\mathrm{sf}}=z_{t-1}^{\\mathrm{sf}}+\\big(y_t^{\\mathrm{sf},+}-y_t^{\\mathrm{sf}}\\big),\\qquad "
                    "x_t^{\\mathrm{sf}}=(1-c_t)x_{t-1}^{\\mathrm{sf}}+c_t z_t^{\\mathrm{sf}}."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Replacing horizon-coupled cosine annealing with causal step-count cooling (`lr=0.014`, "
                    "$r_{\\mathrm{dec}}=0.55$, $S_{\\mathrm{dec}}=100$) and evaluating single-arrival gradients at "
                    "$y_t^{\\mathrm{sf}}$ (`betas=[0, 0.999]`, $\\beta_{\\mathrm{sf}}=0.88$, $k_{\\mathrm{sf}}=4$, "
                    "$c_{\\min}=0.02$) maintains fast binary-mask acquisition across 96-step minimal streams (`dense4`) "
                    "while serving smooth 50-step primal averages $x_t^{\\mathrm{sf}}$ across 480-step image and robotics streams."
                ),
            },
            {
                "label": "Pre-Birth Debiased Unbucketed KRR and Causal Dirichlet-Precision Fisher-Sobolev Calibration",
                "latex": (
                    "\\begin{aligned}"
                    "\\alpha_{\\mathrm{dir},t}&=\\left(1+\\frac{\\max(E_\\nabla(x_t),\\,\\bar E_{\\nabla,t-1})}{\\sigma_\\nabla^2}\\right)^{-1},\\qquad "
                    "w_{\\mathrm{bnd}}(x_t)=1+\\gamma_{\\mathrm{bnd}}(2\\bar x_t-1)^4,\\\\"
                    "\\mathcal S_{\\mathrm{sob}}(\\hat x_G,x_t)&=\\frac12\\sum_{d\\in\\{\\mathrm{row},\\mathrm{col}\\}}"
                    "\\operatorname{mean}\\!\\left(\\frac{0.25}{\\overline{\\hat x_G(1-\\hat x_G)}_d+0.015}\\big(\\nabla_d\\hat x_G-\\nabla_d x_t\\big)^2\\right),\\\\"
                    "\\mathcal L_G^{\\mathrm{dir}}&=\\lambda_{\\mathrm{rec}}\\Big[\\big(1-\\alpha_{\\mathrm{dir},t}+\\alpha_{\\mathrm{dir},t}w_{\\mathrm{bnd}}(x_t)\\big)"
                    "\\operatorname{BCE}(\\hat x_G,x_t)+\\alpha_{\\mathrm{dir},t}w_{\\mathrm{sob}}\\,\\mathcal S_{\\mathrm{sob}}(\\hat x_G,x_t)"
                    "-\\tilde w_{\\mathrm{style},t}g_{t-1}\\sigma_{\\mathrm{sig}}(D)\\Big]+(\\lambda_{\\mathrm{ctx}}+\\alpha_{\\mathrm{dir},t}\\lambda_{\\mathrm{dir,ctx}})\\lVert W_{fc,\\mathrm{ctx}}\\rVert_F^2."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "In `online-gail-bcap-debiased-krr`, per-coordinate RLS covariances $P_{t,r}$ reset cleanly at subspace row birth, "
                    "effective style weight $\\tilde w_{\\mathrm{style},t}=w_{\\mathrm{style}}\\max(g_{\\mathrm{disc},t},\\alpha_{\\mathrm{dir},t})$ "
                    "eliminates single-sample critic drift on continuous unimodal physics while preserving full adversarial crispness on discrete/multimodal "
                    "and constant-field streams, and KRR evaluates up to $C_{\\mathrm{krr}}=512$ unbucketed post-birth continuous observations (`0` stored frames), "
                    "advancing `seq_robotics_fetch_slide` to `5/5`. In `online-gail-bcap-dirichlet-sobolev`, $\\alpha_{\\mathrm{dir},t}=1$ "
                    "on spatially constant fields (`seq_minimal_kernel2x2`, $E_\\nabla(x_t)=0$) and $\\alpha_{\\mathrm{dir},t}<10^{-7}$ on "
                    "all spatially structured streams, suppressing $2\\times 2$ transposed-conv ripple and context-bypass overfitting without "
                    "perturbing any image, token, or robotics task."
                ),
            },
            {
                "label": "Causal Alpha-Beta Barycentric Kinematics, 2D Lattice-Harmonic Lifting, and Z2 Shift-Equivariant Moment Prior",
                "latex": (
                    "\\begin{aligned}"
                    "\\hat p_k&=(\\hat p_{k-1}+\\hat v_{k-1})+\\alpha_{\\mathrm{kf}}\\big(b_k-(\\hat p_{k-1}+\\hat v_{k-1})\\big),\\qquad "
                    "\\hat v_k=\\hat v_{k-1}+\\beta_{\\mathrm{kf}}\\big(b_k-(\\hat p_{k-1}+\\hat v_{k-1})\\big),\\\\"
                    "\\psi(u_x,u_y)&=s_{\\mathrm{wp}}s_f\\big[u_x,\\,u_y,\\,0.5\\sin 2\\pi s_+,\\,0.5\\cos 2\\pi s_-,\\,0.4\\sin 7\\pi s_+,\\,0.4\\cos 7\\pi s_-,\\,0.3\\sin 14\\pi s_+,\\,0.3\\cos 14\\pi s_-\\big]^\\top,\\\\"
                    "\\mathcal M(\\hat x,x)&=(m(\\hat x)-m(x))^2+4\\lVert \\mu(\\hat x)-\\mu(x)\\rVert_2^2+25\\lVert \\sigma^2(\\hat x)-\\sigma^2(x)\\rVert_2^2,\\\\"
                    "\\mathcal L_{\\mathrm{wp}}&=\\mathcal L_{\\mathrm{mom,mse}}(\\hat x_G,x_t)+\\frac{w_{\\mathrm{eq}}}{|\\mathcal S_t|}\\sum_{\\delta\\in\\mathcal S_t}"
                    "\\Big[\\operatorname{BCE}\\big(G(\\phi(\\mathsf T_\\delta x_t),0),\\mathsf T_\\delta x_t\\big)+\\mathcal L_{\\mathrm{mom,mse}}\\big(G(\\phi(\\mathsf T_\\delta x_t),0),\\mathsf T_\\delta x_t\\big)\\Big]."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "In `online-gail-bcap-kinematic-equivariant`, localized mass-conserving wavepackets ($m\\in[0.25,2.5]$, participation ratio "
                    "$\\mathrm{PR}\\in[1,6]$, $x\\ge 0$) initialize exact 2- and 3-point MVUE velocity $\\hat v_2=\\frac12(b_2-b_0)$ and track $k\\ge 3$ "
                    "with $(\\alpha_{\\mathrm{kf}},\\beta_{\\mathrm{kf}})=(0.20,0.04)$ over relative-masked barycenters ($x\\ge 0.015\\max x$), yielding "
                    "identically zero innovation on constant-velocity training streams while damping closed-loop autoregressive rollout jitter $5\\times$. "
                    "With $s_\\pm=u_x\\pm u_y$, lattice-period harmonics $(7\\pi,14\\pi)$ let $G$ compensate its $2\\times 2\\to 4\\times 4\\to 8\\times 8$ "
                    "nearest-upsample block phase, while $N_{\\mathrm{sh}}=6$ exact interior $\\mathbb Z^2$ shifts $\\mathsf T_\\delta x_t$ decouple unseen "
                    "spatial positions from velocity ($8/8$ passing terminal checks on `seq_minimal_velocity_extrapolation`, `0` activation on all other 19 tasks)."
                ),
            },
        ],
        "pseudocode": [
            "Initialize fallback orthogonal encoder E_0, frozen MoG prior P, empty subspace/prototype/context/KRR buffers, P_0 = (1 / assoc_ridge) I, and Schedule-Free state (z_0 = x_0 = theta_G).",
            "For each chronological single-arrival step t (batch_size = 1, replay = none):",
            "  Before revealing x_t, serve y_hat = G_{x_{t-1}}(z_cond(i_serve, c_{t-1}), c_{t-1}) under an isolated eval RNG.",
            "  Reveal single target frame x_t; stage Gram-Schmidt innovation row (or wavepacket harmonic lift if alpha_wp = 1) and encode phi_t(x_t).",
            "  If schedule_free_beta > 0, interpolate generator parameters to y_t = (1 - beta_sf) z_{t-1} + beta_sf x_{t-1}.",
            "  Draw uniform row i_D; form z_D via active-entropy quantiles + debiased RLS / harmonic lift / spike-and-slab + (1 - g_disc) * Matern32_KRR(c_{t-1}); step D with Hinge + BCAP cap.",
            "  With D fixed, draw uniform row i_G and form posterior code z_G = phi_t(x_t) + eps_{i_G}; evaluate survival BCE (+ Dirichlet Fisher-Sobolev or wavepacket Z2 shift-equivariant moment prior if gated) - bounded style bonus + Markov penalty; step G.",
            "  If schedule_free_beta > 0, update dual z_t = z_{t-1} + (y_t^+ - y_t) and primal x_t = (1 - c_t) x_{t-1} + c_t z_t, leaving G at x_t for causal serving.",
            "  After optimizer updates succeed, commit staged innovation row, prototypes, context buckets, unbucketed KRR entry, Dirichlet/wavepacket counters, return baseline, and float64 RLS + covariance.",
            "  Advance detached causal state c_t = CausalStateMemory.write(x_t) (with prev_val readout and optional alpha-beta kinematic tracker) without storing or replaying frames.",
        ],
        "training_details": {
            "loss": "Hinge margin adversarial critic loss (`loss=\"hinge\"`), paired on $G$ with canceled linear critic slope (`cancel_linear_gan=True`), prioritized pixelwise BCE survival anchor ($\\lambda_{\\mathrm{rec}}=10$ in Hinge..SNR; $100$ in Subspace..Kinematic-Equivariant), bounded sigmoid style bonus ($w_{\\mathrm{style}}\\in\\{0.1,0.05,0.002\\}$), Markov context-bypass penalty (`context_bypass_weight=5.0`), causal Dirichlet-precision Fisher-Sobolev $H^1$ + quartic calibration (`Dirichlet-Sobolev` onward), and $\\mathbb Z^2$ shift-equivariant central-moment + $L_2$ wavepacket regularization in `online-gail-bcap-kinematic-equivariant`.",
            "optimizer": "Native Adam via `bcap_adam` (`betas=(0, 0.999)` in Hinge/BCE/Spike-Slab/SNR/Schedule-Free; `(0.5, 0.999)` in Momentum; `(0.4, 0.999)` in Subspace; `(0, 0.95)` paired with polynomial Primal-Dual Schedule-Free $\\beta_{\\mathrm{sf}}=0.88$ in Debiased-KRR, Dirichlet-Sobolev, and Kinematic-Equivariant). Base MoG storage table is frozen (`prior_update=\"frozen\"`).",
            "learning_rates": "Hinge..SNR use `lr=0.005`, `critic_lr_mult=1.0`, cosine-annealed from `0.5` to `0.05`; `online-gail-bcap-subspace` uses `lr=0.007`, `d_lr_mult=0.02`, cosine-annealed from `0.55` to `0.06`; `online-gail-bcap-schedule-free` uses causal `lr_schedule=\"exponential\"` (`lr=0.014`, `lr_decay_rate=0.55`, `lr_decay_steps=100`); Debiased-KRR, Dirichlet-Sobolev, and Kinematic-Equivariant use `lr=0.016`, `lr_decay_rate=0.58`, `lr_decay_steps=100`, `d_lr_mult=0.02`.",
            "gradient_clipping": "None (`critic_grad_guard=False`). Unbounded generator critic gradients are eliminated structurally via the bounded sigmoid style bonus and BCAP input-gradient cap.",
            "critic_penalty": "Fixed one-sided squared $L_2$ BCAP cap on real and conditional-prior fake inputs ($\\lambda=1.0$, $\\kappa=1.0$) every critic step; no EMA critic anchor.",
            "damping_and_guards": "No A2 sparse-row damping (`latent_damping_max_rate=0.0`), zero generator-center probes (`probe_law=\"no_generator_probes_v1\"`), regularized Mahalanobis covariance $\\tilde\\Sigma_t\\succeq 10^{-4}I$, and exact ridge-stabilized Matérn-3/2 KRR ($\\lambda_{\\mathrm{krr}}\\in\\{0.06,0.025\\}$).",
            "online_memory_and_routing": "Detached `CausalStateMemory` ($d_c=16$, `write_lr=0.5` in Hinge..SNR and `0.35` in Subspace..Kinematic-Equivariant, `association_weight=0.0`, `normalize_key=False, key_gain=0.5` from Spike-Slab onward, `contrast_threshold=0.002` from Subspace onward, `prev_val_weight=1.0` from Debiased-KRR onward, `barycentric_kinematics=True` with `kinematic_alpha=0.20, kinematic_beta=0.04` in Kinematic-Equivariant) paired with `GAILConditionalReconstruction` (`router=\"gail_conditional\"`, `policy_entropy_bonus=0.01`, `subspace_dims=6`, `residual_kernel=\"matern32_krr\"`, `krr_capacity=512`, `replay_policy=\"none\"`, `rehearsal_batch=0`).",
            "noise": "Zero scheduled critic-input or generator-output additive noise (`0.0`). Intrinsic task MoG perturbation $\\sigma=0.025$ is preserved on every conditional and posterior latent draw; precision-whitened innovation rows (`subspace_nu0=0.10`, `feature_scale=2.1`) and wavepacket harmonic scaling (`wavepacket_snr_scale=2.5`) amplify continuous physical SNR without clipping discrete binary images.",
            "averaging": "Hinge..Subspace serve live weights (`ema_decay=0.0`, `serve_average=False`); Schedule-Free, Debiased-KRR, Dirichlet-Sobolev, and Kinematic-Equivariant serve the polynomial Primal-Dual Schedule-Free iterate $x_t^{\\mathrm{sf}}$ while interrogating gradients at $y_t^{\\mathrm{sf}}=0.12 z_{t-1}^{\\mathrm{sf}}+0.88 x_{t-1}^{\\mathrm{sf}}$.",
        },
        "configuration_notes": [
            "GAIL-BCAP forms a ten-recipe single-factor ablation ladder over `online-feature-prior-bce`: `online-gail-bcap-hinge` (10/10 required, 1/10 stretch) -> `online-gail-bcap-bce` (10/10 required, 1/10 stretch) -> `online-gail-bcap-momentum` -> `online-gail-bcap-spike-slab` (10/10 required, 3/10 stretch) -> `online-gail-bcap-snr` (10/10 required, 3/10 stretch) -> `online-gail-bcap-subspace` (10/10 required, 4/10 stretch) -> `online-gail-bcap-schedule-free` (10/10 required, 5/10 stretch) -> `online-gail-bcap-debiased-krr` (10/10 required, 6/10 stretch) -> `online-gail-bcap-dirichlet-sobolev` (10/10 required, 7/10 stretch) -> `online-gail-bcap-kinematic-equivariant` (10/10 required, 8/10 stretch).",
            "`online-gail-bcap-subspace` upgrades `online-gail-bcap-snr` with precision-whitened anchored Gram-Schmidt innovation subspaces (`subspace_dims=6`), `float64` Matérn-3/2 KRR over per-bucket RLS residuals (`residual_kernel=\"matern32_krr\"`), fine-salience action gating (`contrast_threshold=0.002`), and calmed Hinge-BCAP calibration (`d_lr_mult=0.02`, `reconstruction_weight=100.0`, `style_bonus_weight=0.002`), passing `seq_robotics_fetch_pick_and_place` (`5/5`).",
            "`online-gail-bcap-schedule-free` upgrades `online-gail-bcap-subspace` with causal exponential cooling (`lr_schedule=\"exponential\"`, `lr=0.014`, `lr_decay_rate=0.55`) and polynomial Primal-Dual Schedule-Free averaging (`betas=[0.0, 0.999]`, `schedule_free_beta=0.88`, `schedule_free_poly_k=4.0`, `schedule_free_c_min=0.02`), reaching `15/20` (`+ seq_minimal_dense4` `5/5`).",
            "`online-gail-bcap-debiased-krr` upgrades `online-gail-bcap-schedule-free` with orthogonal previous-frame value readout (`prev_val_weight=1.0`), pre-birth subspace RLS/KRR zero-debiasing (`subspace_birth_debias=True`, `subspace_rel_tol=0.055`, `assoc_ridge=0.08`), unbucketed 512-point Matérn-3/2 KRR (`krr_capacity=512`, `residual_bandwidth=0.72`, `residual_ridge=0.025`), consistent discrete-fork control-variate branch anchoring (`consistent_fork_branch=True`), discreteness-gated style bonus (`disc_gate_style=True`), and short-horizon Adam $\\beta_2=0.95$ Schedule-Free cooling (`lr=0.016`, `lr_decay_rate=0.58`, `schedule_free_poly_k=2.4`, `schedule_free_c_min=0.005`), reaching `16/20` (`+ seq_robotics_fetch_slide` `5/5`).",
            "`online-gail-bcap-dirichlet-sobolev` is a strict single-factor child of `online-gail-bcap-debiased-krr` adding causal Empirical-Bayes Dirichlet-precision gated Fisher-Sobolev $H^1$ regularization (`sobolev_weight=36.0`, `sobolev_sigma=1e-7`), quartic boundary logit calibration (`boundary_quartic_gain=0.85`), Dirichlet Markov bottleneck shrinkage (`dirichlet_bypass_weight=55.0`), and Schedule-Free polynomial/floor boost (`dirichlet_c_boost=0.02`), reaching `17/20` (`+ seq_minimal_kernel2x2` `5/5`).",
            "`online-gail-bcap-kinematic-equivariant` is a strict single-factor child of `online-gail-bcap-dirichlet-sobolev` adding causal $\\alpha$-$\\beta$ barycentric kinematic tracking (`barycentric_kinematics=True`, `kinematic_alpha=0.2`, `kinematic_beta=0.04`), 2D lattice-harmonic barycentric latent lifting (`wavepacket_snr_scale=2.5`), and $\\mathbb Z^2$ shift-equivariant central-moment + $L_2$ decoder calibration (`wavepacket_equiv_weight=3.0`, `wavepacket_moment_weight=0.07`, `wavepacket_mse_weight=2.5`, `wavepacket_bypass_weight=80.0`, `wavepacket_shifts=6`, `wavepacket_c_boost=0.205`), ranking #1 overall on `single_stream_stability` view 10 at `18/20` (`10/10` required + `8/10` stretch: conquering Tier 3 `seq_minimal_velocity_extrapolation` `5/5` alongside all 4 Tier 2 minimal tasks and 3 Tier 3 Fetch robotics tasks).",
            "Executed strictly with `batch_size=1` (real and fake), `seed=0`, `replay_policy=\"none\"`, `rehearsal_batch=0`, and zero generator probes across all 20 tasks.",
        ],
        "sources": [
            "particlestream/gail_prior.py",
            "particlestream/transport.py",
            "particlestream/feature_prior.py",
            "particlestream/memory.py",
            "particlestream/tier1.py",
            "configs/forge/ideas/online-gail-bcap-hinge.json",
            "configs/forge/ideas/online-gail-bcap-bce.json",
            "configs/forge/ideas/online-gail-bcap-momentum.json",
            "configs/forge/ideas/online-gail-bcap-spike-slab.json",
            "configs/forge/ideas/online-gail-bcap-snr.json",
            "configs/forge/ideas/online-gail-bcap-subspace.json",
            "configs/forge/ideas/online-gail-bcap-schedule-free.json",
            "configs/forge/ideas/online-gail-bcap-debiased-krr.json",
            "configs/forge/ideas/online-gail-bcap-dirichlet-sobolev.json",
            "configs/forge/ideas/online-gail-bcap-kinematic-equivariant.json",
        ],
        "references": [
            {
                "title": "Generative Adversarial Imitation Learning",
                "url": "https://arxiv.org/abs/1606.03476",
                "component": "Imitation-style occupancy matching with causal return advantage and policy entropy regularization.",
            },
            {
                "title": "Geometric GAN",
                "url": "https://arxiv.org/abs/1705.02894",
                "component": "Hinge margin critic objective paired with the BCAP input-gradient cap.",
            },
            {
                "title": "High-Dimensional Continuous Control Using Generalized Advantage Estimation",
                "url": "https://arxiv.org/abs/1506.02438",
                "component": "Discounted return baseline and variance-standardized advantage gain for online policy updates.",
            },
            {
                "title": "Bayesian Variable Selection in Linear Regression",
                "url": "https://doi.org/10.1080/01621459.1988.10478693",
                "component": "Spike-and-slab Bayes factor formulation powering Mahalanobis discrete-vs-continuous prototype gating.",
            },
            {
                "title": "Numerics of Gram-Schmidt Orthogonalization",
                "url": "https://doi.org/10.1016/0024-3795(94)90493-6",
                "component": "Two-pass classical Gram-Schmidt reorthogonalization anchoring precision-whitened online innovation directions.",
            },
            {
                "title": "The Road Less Scheduled",
                "url": "https://arxiv.org/abs/2405.15682",
                "component": "Primal-Dual Schedule-Free iterate interpolation (y_t, z_t, x_t) paired with causal horizon-invariant step cooling.",
            },
        ],
    },
    "BCAP": {
        "overview": (
            "BCAP regularizes the critic with a fixed one-sided squared hinge penalty whenever per-sample "
            "real or generated input-gradient norms exceed cap threshold $\\kappa$. Within ParticleStream, "
            "this critic-cap formulation spans DualNorm and Adam optimizers, unrouted uniform particle "
            "sampling, competitive 32-center MSE routing, and context-balanced Kantorovich dual routing "
            "paired with detached velocity or causal-state memory under `batch_size=1` online execution."
        ),
        "symbols": (
            "$G$ is the generator, $D$ the raw-score critic, and $P$ the $K$-component task prior ($K=32$, "
            "$d_z=8$, $\\sigma=0.025$) with centers $\\{\\mu_j\\}_{j=0}^{K-1}$. At step $t$, $x_t$ is the "
            "single real arrival, $c_{t-1}$ is the optional detached memory context, $\\hat x_D=G(z_{i_D},c_{t-1})$ "
            "is the detached critic fake, and $\\hat x_G=G(z_{i_G},c_{t-1})$ is the differentiable generator fake. "
            "$g_r=\\nabla_x D(x_t,c_{t-1})$ and $g_f=\\nabla_{\\hat x} D(\\hat x_D,c_{t-1})$ are per-sample input "
            "gradients; $\\lambda$ is penalty strength, $\\kappa$ is the gradient cap, and $(a)_+=\\max(a,0)$. "
            "For routed variants, $\\hat x_j^{(0)}=G(\\mu_j,c_{t-1})$ denotes a detached eval-mode clean-center "
            "probe, $m_j$ is global assignment mass ($\\gamma_u$ decay, $\\lambda_u$ usage weight), and "
            "$u_b\\in\\mathbb R^K$ is the zero-sum clipped Kantorovich potential vector at matched context bucket $b$."
        ),
        "equations": [
            {
                "label": "Fixed Capped Input-Gradient Penalty",
                "latex": (
                    "R_{\\mathrm{BCAP}}(D;x_t,\\hat x_D)=\\frac{\\lambda}{2}\\Big["
                    "(\\lVert g_r\\rVert_2-\\kappa)_+^2+(\\lVert g_f\\rVert_2-\\kappa)_+^2\\Big]."
                ),
                "explanation": (
                    "BCAP penalizes critic input-gradient norms above $\\kappa$ without dividing by input "
                    "dimension $d=64$ and without hard-clipping critic parameter gradients. Canonical "
                    "ParticleStream candidates use $\\lambda=1$ and $\\kappa=1$."
                ),
            },
            {
                "label": "Single-Arrival Critic and Generator Objectives",
                "latex": (
                    "\\begin{aligned}"
                    "\\mathcal L_D&=\\ell_D\\big(D(x_t,c_{t-1}),\\,D(\\hat x_D,c_{t-1})\\big)+R_{\\mathrm{BCAP}}(D;x_t,\\hat x_D),\\\\"
                    "\\mathcal L_G&=\\ell_G\\big(D(\\hat x_G,c_{t-1}),\\,D(x_t,c_{t-1})\\big)+R_{\\mathrm{prior}}(P)+"
                    "\\lambda_{\\mathrm{rec}}\\,\\mathcal L_{\\mathrm{rec}}(\\hat x_G,x_t)."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Preset candidates inherit paired relativistic logistic $\\ell_D,\\ell_G$, whereas tuned "
                    "online candidates switch to decoupled non-saturating logistic loss so single-sample "
                    "`batch_size=1` mode mismatches do not invert real-vs-fake score differences."
                ),
            },
            {
                "label": "Competitive and Context-Balanced Particle Routing",
                "latex": (
                    "\\begin{aligned}"
                    "s_j^{\\mathrm{comp}}&=\\frac{1}{64}\\lVert \\hat x_j^{(0)}-x_t\\rVert_F^2+\\lambda_u\\log(1+m_j),\\qquad "
                    "s_j^{\\mathrm{bal}}=\\frac{1}{64}\\lVert \\hat x_j^{(0)}-x_t\\rVert_F^2+u_{b,j},\\\\"
                    "\\tilde u_b&=u_b+\\eta_{\\mathrm{dual}}\\!\\left(e_{i_G}-\\frac{1}{K}\\mathbf 1_K\\right),\\qquad "
                    "u_b\\leftarrow\\big(\\tilde u_b-\\operatorname{mean}(\\tilde u_b)\\mathbf 1_K\\big)\\cdot"
                    "\\min\\!\\left(1,\\,\\frac{u_{\\mathrm{clip}}}{\\lVert \\tilde u_b-\\operatorname{mean}(\\tilde u_b)\\mathbf 1_K\\rVert_\\infty}\\right)."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Critic steps always sample $i_D\\sim\\mathcal U\\{0,\\dots,K-1\\}$ uniformly so serving "
                    "matches the uniform prior. Generator steps either sample uniformly (`router=None`) or "
                    "probe all $K=32$ clean centers in eval mode and select $i_G=\\arg\\min_j s_j$ (or softmax "
                    "at temperature $T>0$), maintaining either global load-balancing mass $m$ or per-context "
                    "zero-sum duals $u_b$."
                ),
            },
        ],
        "pseudocode": [
            "Choose optimizer (DualNorm or Adam), adversarial loss, memory reader and optional particle router.",
            "For each chronological single-arrival step t:",
            "  Serve pre-arrival prediction from uniform prior row under isolated eval RNG when task is predictive.",
            "  Sample uniform critic index i_D ~ Uniform({0, ..., K - 1}) and form detached fake x_hat_D = G(z_{i_D}, c_{t-1}).",
            "  Compute input gradients g_r = grad_x D(x_t, c_{t-1}) and g_f = grad_x D(x_hat_D, c_{t-1}).",
            "  Backpropagate ell_D + fixed BCAP cap penalty and step D.",
            "  If a router is configured, evaluate K = 32 detached clean-center probes G(mu_j, c_{t-1}) against x_t",
            "    and select i_G using MSE + global usage penalty or context-local Kantorovich duals u_b; else draw uniform i_G.",
            "  Recompute differentiable fake x_hat_G = G(z_{i_G}, c_{t-1}) and evaluate ell_G + R_prior + lambda_rec L_rec(x_hat_G, x_t).",
            "  Step G (and trainable prior P unless prior_update = 'frozen'); update dual/usage state and detached memory c_t.",
        ],
        "training_details": {
            "loss": "Paired relativistic logistic in preset baselines (`online-bcap`, `online-bcap-adam`); decoupled non-saturating logistic in tuned online recipes, optionally paired with MSE or weight-10 BCE reconstruction on routed generator steps.",
            "optimizer": "`bcap` preset uses DualNorm (`online-bcap`); `bcap_adam` recipes use Adam ($\\beta_1=0$, $\\beta_2=0.999$ default or $0.99$ in tuned online recipes).",
            "learning_rates": "Preset baselines run constant LRs (`0.012` DualNorm, `0.00425` Adam); tuned online recipes use base LR `0.005` cosine-annealed from $50\\%$ of budget to `lr_floor=0.05`.",
            "gradient_clipping": "None (`critic_grad_guard=False`). Critic regularity comes solely from the soft BCAP input-gradient hinge.",
            "critic_penalty": "Fixed real/fake one-sided squared $L_2$ cap ($\\lambda=1.0$, $\\kappa=1.0$) every critic step; no EMA critic anchor.",
            "damping_and_guards": "No A2 sparse-row damping (`latent_damping_max_rate=0.0`). Frozen-prior variants (`online-bcap-balanced-frozen`, `online-bcap-balanced-bce`) eliminate sparse-row Adam drift altogether.",
            "online_memory_and_routing": "Supports `kind=\"none\"`, `FastWeightMemory` (`reader=\"velocity\"`), or `CausalStateMemory` (`reader=\"causal_state\"`), combined with uniform, `competitive`, or `context_balanced` routing (`replay_policy=\"none\"`, `rehearsal_batch=0` on all active recipes).",
            "noise": "Zero scheduled critic-input or generator-output noise in tuned online recipes; task-owned MoG $\\sigma=0.025$ is preserved.",
            "averaging": "Live weights (`ema_decay=0.0`, `serve_average=False`) on tuned online candidates.",
        },
        "configuration_notes": [
            "The fixed BCAP critic input-gradient cap defines the family; optimizer choice, loss pairing, prior freeze, memory reader, and generator routing are explicit recipe deltas.",
            "Among BCAP recipes, `online-bcap-balanced-bce` is the family representative (`3/10` required passes, `1/10` stretch pass on `seq_minimal_tokens4`), demonstrating that context-local zero-sum duals + BCE reconstruction capture stationary modes and 1-step motion (`bars4`, `dwell_bars4`, `translating_bar_next`) but still diffuse uniform conditional capacity across contexts.",
            "Historical `bcap`, `stream-bcap`, and `stream-bcap-adam` declarations predate the explicit `replay_policy=\"none\"` schema and remain read-only provenance records.",
        ],
        "sources": [
            "particlestream/tier1.py",
            "particlestream/transport.py",
            "particlestream/balanced_transport.py",
            "particlestream/memory.py",
            "configs/forge/ideas/online-bcap-balanced-bce.json",
            "configs/forge/ideas/online-bcap-adam-causal.json",
        ],
        "references": [
            {
                "title": "On the regularization of Wasserstein GANs",
                "url": "https://arxiv.org/abs/1709.08894",
                "component": "Related one-sided squared hinge gradient penalty; BCAP applies the cap directly at real and fake samples.",
            },
            {
                "title": "The relativistic discriminator: a key element missing from standard GAN",
                "url": "https://arxiv.org/abs/1807.00734",
                "component": "Paired relativistic logistic loss used in preset `bcap` and `bcap_adam` baselines.",
            },
            {
                "title": "Generative Adversarial Networks",
                "url": "https://arxiv.org/abs/1406.2661",
                "component": "Non-saturating logistic adversarial loss used in tuned `batch_size=1` online BCAP recipes.",
            },
            {
                "title": "Adam: A Method for Stochastic Optimization",
                "url": "https://arxiv.org/abs/1412.6980",
                "component": "Base optimizer in `bcap_adam` recipes.",
            },
        ],
    },
    "BISPF": {
        "overview": (
            "Born-Infeld Sinkhorn Particle Flow (BISPF) is a critic-regularized ParticleGAN formulation "
            "without an EMA critic copy. It couples real and generated batches via scale-free entropic "
            "Optimal Transport (Sinkhorn-Knopp), bounds critic slopes with a two-regime Elastic Born-Infeld "
            "potential and a zero-mean logit gauge, and provides optional Stratified Blue-Noise Particle "
            "(SBNP) prior regularization and Harmonic Mode Capture thermal cooling. Under ParticleStream's "
            "`batch_size=1` sequential contract, the Sinkhorn coupling reduces to a single $(1,1)$ pair "
            "while retaining EMA residual/scale cap adaptation."
        ),
        "symbols": (
            "$G$ is the generator, $D$ the raw-score critic, $P$ the $K$-particle task prior ($K=32$, "
            "$d_z=8$, $\\sigma=0.025$), $x_t$ the single real arrival, $c_{t-1}$ the detached memory context, "
            "and $y_t=G(z_i,c_{t-1})$ the generated sample. $\\Pi_\\varepsilon$ is the Sinkhorn coupling at "
            "entropic scale $\\varepsilon=0.15$ ($\\Pi_\\varepsilon=[1]$ when $B_r=B_f=1$), with Sinkhorn "
            "marginal-residual EMA $\\bar r_t\\in[0,1]$ and cloud-scale EMA $\\bar s_t>0$ (decay $\\gamma_{\\mathrm{th}}=0.95$). "
            "$g_r=\\nabla_x D(x_t,c_{t-1})$, $g_f=\\nabla_y D(y_t,c_{t-1})$, and $g_c=\\nabla_{x_{\\mathrm{chord}}} D(x_{\\mathrm{chord}},c_{t-1})$ "
            "are critic input gradients; $d=64$ counts $1\\times 8\\times 8$ image coordinates, $\\kappa$ is "
            "the nominal cap, $\\phi_\\beta(u)=\\beta^{-1}\\log(1+e^{\\beta u})$ ($\\beta=30$) is the smooth "
            "Born-Infeld wall, $\\lambda_{\\mathrm{floor}}$ is the harmonic floor weight, $\\lambda_{\\mathrm{cap}}$ "
            "the elastic cap weight, $\\alpha_c$ the chord blend, and $\\gamma_{\\mathrm{gauge}}$ the zero-mean "
            "logit gauge weight."
        ),
        "equations": [
            {
                "label": "Elastic Born-Infeld Regularizer and Zero-Mean Logit Gauge",
                "latex": (
                    "R_{\\mathrm{BISPF}}=\\frac{\\lambda_{\\mathrm{floor}}}{2 m_g d}\\,\\mathbb E\\!\\left["
                    "\\sum_{g\\in\\mathcal G}\\lVert g\\rVert_2^2\\right]+"
                    "\\frac{\\lambda_{\\mathrm{cap}}}{2}\\,\\mathbb E\\!\\left["
                    "\\phi_\\beta(\\lVert g_r\\rVert_2-\\kappa_{\\mathrm{eff}})^2+"
                    "(1-\\alpha_c)\\phi_\\beta(\\lVert g_f\\rVert_2-\\kappa_{\\mathrm{eff}})^2+"
                    "\\alpha_c\\phi_\\beta(\\lVert g_c\\rVert_2-\\kappa_{\\mathrm{eff}})^2\\right]+"
                    "\\frac{\\gamma_{\\mathrm{gauge}}}{2}\\bar D^2."
                ),
                "explanation": (
                    "Here $\\bar D=\\frac12(\\mathbb E[D(x_t,c_{t-1})]+\\mathbb E[D(y_t,c_{t-1})])$ pins the "
                    "translational logit gauge without an EMA critic network. On 4D image tensors at default "
                    "$\\alpha_c=0.02\\le 0.10$, the BISPF image gate bypasses pixel-chord interpolation "
                    "($\\mathcal G=\\{g_r,g_f\\}$, $m_g=2$, effectively $\\alpha_c=0$ in the cap) so discrete "
                    "spatial modes are not blended across pixel space, using $\\lambda_{\\mathrm{floor}}^{\\mathrm{img}}=5\\times 10^{-4}$ "
                    "and $\\gamma_{\\mathrm{gauge}}^{\\mathrm{img}}=5\\times 10^{-3}$."
                ),
            },
            {
                "label": "Scale- and Residual-Adapted Image Cap",
                "latex": (
                    "s_{\\mathrm{img}}=\\operatorname{clip}\\!\\left(\\frac{\\bar s_t}{s_{\\mathrm{ref}}},\\,0.15,\\,1.0\\right),\\qquad "
                    "\\kappa_{\\mathrm{eff}}=\\frac{\\kappa}{s_{\\mathrm{img}}}\\Big(0.6+1.1\\big(1-\\operatorname{clip}(\\bar r_t,0,1)\\big)\\Big)."
                ),
                "explanation": (
                    "With $s_{\\mathrm{ref}}=4.899$, the effective slope wall $\\kappa_{\\mathrm{eff}}$ adapts "
                    "smoothly from the observed Sinkhorn scale EMA $\\bar s_t$ and residual EMA $\\bar r_t$ "
                    "even when each online arrival has `batch_size=1`."
                ),
            },
            {
                "label": "Stratified Blue-Noise Particle (SBNP) Prior Regularizer",
                "latex": (
                    "R_{\\mathrm{SBNP}}(P)=\\lambda_P\\left(\\mathcal T_{\\mathrm{moment}}(P)+w_{\\mathrm{repel}}\\,\\mathbb I(K>2k_{\\mathrm{nn}}+d_z)\\,"
                    "\\frac{1}{K k_{\\mathrm{nn}}}\\sum_{i=1}^K\\sum_{m=1}^{k_{\\mathrm{nn}}}\\log\\!\\left(1+\\frac{\\bar r_{\\mathrm{knn}}^2}{\\lVert \\mu_i-\\mu_{\\mathcal N_m(i)}\\rVert_2^2+\\epsilon}\\right)\\right)."
                ),
                "explanation": (
                    "The `bispf` and `bispf_optimized` presets enable SBNP (`prior_regularizer=\"sbnp\"`), "
                    "whereas ParticleStream's tuned online BISPF recipes override `prior_regularizer=\"none\"` "
                    "(and `prior_update=\"frozen\"` in `online-bispf-frozen`) so 32 particles can represent "
                    "4 discrete visual modes without repulsive blue-noise spreading."
                ),
            },
        ],
        "pseudocode": [
            "Choose optimizer (DualNorm in bispf_optimized or BISPFCriticAdam in bispf), rates and prior policy.",
            "For each chronological single-arrival step t:",
            "  Serve pre-arrival prediction under isolated eval RNG when task requires causal prediction.",
            "  Draw particle row i_D and compute detached fake y_D = stop_gradient(G(z_{i_D}, c_{t-1})).",
            "  Evaluate single-pair Sinkhorn coupling Pi = [[1]] between x_t and y_D; update residual and scale EMAs.",
            "  Compute EBI harmonic floor + smooth softplus_beta cap at kappa_eff on input gradients plus zero-mean logit gauge.",
            "  Backpropagate critic non-saturating logistic loss + BISPF penalty and step D.",
            "  With D fixed, draw particle row i_G (uniform or routed), recompute differentiable fake y_G = G(z_{i_G}, c_{t-1}),",
            "    and backpropagate generator adversarial loss (+ declared prior or reconstruction terms).",
            "  Step G (and trainable prior P when unfrozen) and advance detached memory c_t = memory.write(x_t).",
        ],
        "training_details": {
            "loss": "Decoupled non-saturating logistic loss (`loss=\"non_saturating\"`) across declared ParticleStream BISPF candidates; routed causal variant (`online-bispf-transport-causal`) adds MSE reconstruction.",
            "optimizer": "`bispf` preset uses `BISPFCriticAdam` / `K3PGeneratorAdam` ($\\beta_1=0$, $\\beta_2=0.99$ in tuned candidates); `bispf_optimized` uses DualNorm (`optimizer_svd_backend=\"batched\"`).",
            "learning_rates": "Preset `bispf` / `bispf_optimized` run constant high rates (`0.008` / `0.012`, `critic_lr_mult=1.5`, `prior_lr_mult=2.5`); tuned online BISPF candidates use `lr=0.005` cosine-annealed to `lr_floor=0.05` (with `lr_anneal_start=0.25` in `online-bispf-early*` and `online-bispf-frozen`).",
            "gradient_clipping": "None (`critic_grad_guard=False`). Bounding acts through the smooth $\\beta=30$ Elastic Born-Infeld potential.",
            "critic_penalty": "Elastic Born-Infeld critic regularizer ($\\kappa=1.0$, $\\beta=30$, image floor $5\\times 10^{-4}$) plus zero-mean logit gauge ($5\\times 10^{-3}$ on images). Zero EMA critic anchor (`reg_anchor_weight=0.0`).",
            "damping_and_guards": "Harmonic Mode Capture thermal cooler (`thermal_cooling=False` in `bispf`, enabled or `bispf_cooler_freeze_2d` in `bispf_optimized*`). No A2 row damping.",
            "online_memory_and_routing": "`FastWeightMemory` velocity reader on most BISPF candidates; `CausalStateMemory` + competitive transport on `online-bispf-transport-causal`. All active candidates enforce `replay_policy=\"none\"` and `rehearsal_batch=0`.",
            "noise": "Zero scheduled critic-input or generator-output noise (`0.0`); task-declared MoG $\\sigma=0.025$ is preserved.",
            "averaging": "Live unaveraged weights (`ema_decay=0.0`, `serve_average=False`).",
        },
        "configuration_notes": [
            "At `batch_size=1`, Sinkhorn coupling between one real and one generated sample is always the $1\\times 1$ matrix $[1]$; BISPF therefore acts purely through its scale-adapted EBI gradient wall and zero-mean logit gauge rather than multi-sample OT pairing.",
            "`online-bispf-frozen` is the strongest measured BISPF recipe (`1/10` required passes, passing `seq_img_round_robin_bars4`), confirming that freezing the 32-row MoG table prevents sparse-row drift on i.i.d. cycling.",
            "Preset `online-bispf-optimized*` recipes inherit `lr=0.012`, `critic_lr_mult=1.5`, and repulsive `sbnp`, which destabilize single-sample $8\\times 8$ streams (`DEAD_ENDS.md`).",
        ],
        "sources": [
            "particlestream/tier1.py",
            "particlestream/transport.py",
            "particlestream/memory.py",
            "configs/forge/ideas/online-bispf-frozen.json",
            "configs/forge/ideas/online-bispf-early.json",
            "configs/forge/ideas/online-bispf-transport-causal.json",
        ],
        "references": [
            {
                "title": "Sinkhorn Distances: Lightspeed Computation of Optimal Transport",
                "url": "https://arxiv.org/abs/1306.0895",
                "component": "Entropic Optimal Transport Sinkhorn-Knopp coupling and marginal-residual statistics.",
            },
            {
                "title": "Adam: A Method for Stochastic Optimization",
                "url": "https://arxiv.org/abs/1412.6980",
                "component": "First-order moment optimizer used in `bispf` recipes alongside DualNorm in `bispf_optimized`.",
            },
            {
                "title": "The relativistic discriminator: a key element missing from standard GAN",
                "url": "https://arxiv.org/abs/1807.00734",
                "component": "Relativistic pairing foundation supported by the upstream BISPF preset family.",
            },
        ],
    },
    "KA2": {
        "overview": (
            "KA2 pairs a critic-local Adam controller (`KA2CriticAdam`) with a readiness-gated two-phase "
            "input-gradient penalty and an adaptive EMA critic-gradient anchor. Before critic second-moment "
            "readiness crosses its threshold, KA2 applies an RMS-scaled warm-up penalty; once ready, it "
            "blends the warm-up term equally with an unnormalized $L_2$ cap and an EMA real-gradient "
            "proximity anchor whose release gate and tracking speed respond to Adam moment surprise."
        ),
        "symbols": (
            "$G$ is the generator, $D$ the live critic with parameters $\\theta_t$, $\\bar D$ the critic "
            "parameter EMA with parameters $\\bar\\theta_t$, and $C,\\bar C$ denote $D,\\bar D$ wrapped with "
            "any declared fresh critic-input noise. $x_t$ is the single real arrival, $c_{t-1}$ the detached "
            "memory context, and $y_t$ the generated sample (including any scheduled output noise). "
            "$g_r=\\nabla_x C(x_t,c_{t-1})$ and $g_f=\\nabla_y C(y_t,c_{t-1})$ are per-sample input gradients "
            "across $d=64$ coordinates; $\\lambda$ is penalty strength, $\\kappa$ is the cap threshold, and "
            "$w$ is configured anchor weight (`reg_anchor_weight`). $R(t)=1-\\beta_{2,D}^t$ is critic Adam "
            "second-moment readiness against threshold $R_{\\mathrm{thresh}}=0.55085\\approx 1-0.999^{800}$, "
            "$W_t\\in\\{0,1\\}$ is the hysteresis anchor gate driven by short-to-baseline moment-surprise "
            "ratio $\\rho_t$, and $\\alpha_t\\in[0,1]$ controls critic EMA decay $\\delta_t=1-0.1\\alpha_t$."
        ),
        "equations": [
            {
                "label": "Early RMS, Late Capped and EMA Anchor Terms",
                "latex": (
                    "\\begin{aligned}"
                    "A&=\\frac{\\lVert g_r\\rVert_2^2}{d}+\\left(\\frac{\\lVert g_f\\rVert_2}{\\sqrt d}-\\kappa\\right)_+^2,\\\\"
                    "B_{\\mathrm{cap}}&=(\\lVert g_r\\rVert_2-\\kappa)_+^2+(\\lVert g_f\\rVert_2-\\kappa)_+^2,\\qquad "
                    "P_{\\mathrm{anchor}}=\\frac{\\lVert g_r-\\operatorname{stopgrad}(\\nabla_x\\bar C(x_t,c_{t-1}))\\rVert_2^2}{d}."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "$A$ normalizes gradient magnitudes by $d=64$ ($\\sqrt d=8$), whereas $B_{\\mathrm{cap}}$ "
                    "caps raw Euclidean norms directly. On the first blended call, $\\bar D$ copies $D$ and "
                    "$P_{\\mathrm{anchor}}=0$; subsequent calls penalize real-gradient deviation from $\\bar C$."
                ),
            },
            {
                "label": "Moment-Readiness Handover",
                "latex": (
                    "R_{\\mathrm{KA2}}=\\begin{cases}"
                    "\\dfrac{\\lambda}{2}A,&R(t)<R_{\\mathrm{thresh}},\\\\[8pt]"
                    "\\dfrac{\\lambda}{2}\\left[\\dfrac12 A+\\dfrac12\\left(B_{\\mathrm{cap}}+W_t w P_{\\mathrm{anchor}}\\right)\\right],&R(t)\\ge R_{\\mathrm{thresh}}."
                    "\\end{cases}"
                ),
                "explanation": (
                    "In `particlegan==0.9.7`, handover triggers when $R(t)=1-\\beta_{2,D}^t\\ge 0.55085$. "
                    "At preset $\\beta_{2,D}=0.999$, a 480-step task reaches $R(480)\\approx 0.381<0.55085$ "
                    "and stays in warm-up $A$; `online-ka2-ready*` sets $\\beta_{2,D}=0.99$ so blending "
                    "engages at step $t=80$ ($R(80)\\approx 0.5525$)."
                ),
            },
            {
                "label": "Surprise-Gated Hysteresis and Adaptive Critic EMA",
                "latex": (
                    "W_t=\\begin{cases}0,&W_{t-1}=1\\land\\rho_t>3.0,\\\\1,&W_{t-1}=0\\land\\rho_t<1.75,\\\\W_{t-1},&\\text{otherwise},\\end{cases}"
                    "\\qquad \\delta_t=1-0.1\\alpha_t,\\qquad \\bar\\theta_t=\\delta_t\\bar\\theta_{t-1}+(1-\\delta_t)\\theta_t."
                ),
                "explanation": (
                    "Once 25 blended calls establish baseline Adam moment surprise, $\\rho_t>3$ releases "
                    "the anchor ($W_t=0$) and speeds up $\\alpha_t$, while $\\rho_t<1.75$ restores $W_t=1$. "
                    "Sixty consecutive high-surprise steps hard-reseed $\\bar\\theta_t\\leftarrow\\theta_t$."
                ),
            },
        ],
        "pseudocode": [
            "Configure KA2CriticAdam, K3PGeneratorAdam, noise schedules and critic moment factor beta_{2,D}.",
            "For each chronological single-arrival step t:",
            "  Serve pre-arrival prediction under isolated eval RNG when task is predictive.",
            "  Draw particle row i_D and form detached fake y_D = stop_gradient(G(z_{i_D}, c_{t-1}) + gen_noise).",
            "  Evaluate input gradients g_r and g_f on noise-wrapped critic C; compute early RMS term A.",
            "  If 1 - beta_{2,D}^t < 0.55085, set R_KA2 = 0.5 * lambda * A.",
            "  Otherwise, initialize or query D_bar for P_anchor, update hysteresis gate W_t from surprise ratio rho_t,",
            "    and set R_KA2 = 0.5 * lambda * (0.5 * A + 0.5 * (B_cap + W_t * w * P_anchor)).",
            "  Backpropagate critic loss + R_KA2, apply spike guard after warmup, step D, and update/reseed D_bar from moment surprise.",
            "  With D fixed, draw i_G (uniform or routed), recompute y_G, apply A2 sparse-row damping if enabled, step G/P, and write c_t.",
        ],
        "training_details": {
            "loss": "Paired relativistic logistic in preset `online-ka2`; decoupled non-saturating logistic in `online-ka2-clean`, `online-ka2-light-anchor`, `online-ka2-ready*`, and `online-ka2-transport-causal`.",
            "optimizer": "`KA2CriticAdam` and `K3PGeneratorAdam` ($\\beta_1=0$; $\\beta_{2,D}=0.999$ in `online-ka2*` vs $0.99$ in `online-ka2-ready*` and `online-ka2-transport-causal`).",
            "learning_rates": "Preset `online-ka2` uses `lr=0.00425` (`lr_anneal_start=0.6`, `lr_floor=0.01`); clean/ready online variants use `lr=0.005` (`lr_anneal_start=0.5`, `lr_floor=0.05`, `prior_lr_mult=2.0` or `1.0`).",
            "gradient_clipping": "Adaptive per-tensor critic spike guard (`critic_grad_guard=True` in preset, disabled in `online-ka2-ready*`) limits tensor RMS to $5\\times$ bias-corrected second-moment RMS after 200 steps.",
            "critic_penalty": "Readiness-gated blend of early RMS penalty $A$ and late $L_2$ cap $B_{\\mathrm{cap}}$ plus surprise-gated EMA gradient anchor ($w=1.0$ default, $w=0.1$ in `light-anchor`, $w=0.25$ in `ready*`).",
            "damping_and_guards": "A2 sparse latent-row cosine damping (`latent_damping_max_rate=0.5` in preset, disabled in `online-ka2-ready*`).",
            "online_memory_and_routing": "`FastWeightMemory` velocity reader (`online-ka2*`), `kind=\"none\"` (`online-ka2-stateless`), or `CausalStateMemory` (`online-ka2-ready-causal`, `online-ka2-transport-causal`); all active recipes enforce `replay_policy=\"none\"` and `rehearsal_batch=0`.",
            "noise": "Preset `online-ka2` applies decaying critic-input noise ($0.5\\to 0$) and generator output noise ($0\\to 0.029$); `clean` and `ready*` variants set both to $0.0$ for crisp binary $8\\times 8$ gates.",
            "averaging": "Critic EMA $\\bar D$ adapts in $\\delta_t\\in[0.9,1.0]$ once blended; generator EMA (`0.995` in preset, `0.0` in `clean`/`ready*`) is not used for serving (`serve_average=False`).",
        },
        "configuration_notes": [
            "All measured strict-online KA2 candidates currently score `0/10` required passes (`online-ka2-transport-causal` reaches `1/10` stretch pass on `seq_minimal_tokens4`).",
            "On 480-step online streams, preset $\\beta_{2,D}=0.999$ never reaches blended mode ($R(480)<0.55085$), while $\\beta_{2,D}=0.99$ (`online-ka2-ready`) activates the EMA critic anchor across switching regimes where anchoring to stale real-input gradients slows online adaptation.",
            "Preset additive critic-input noise (`0.5`) and generator output noise (`0.029`) blur sharp binary $8\\times 8$ bars below pixel-quality thresholds, motivating zero-noise overrides in `online-ka2-clean` and `online-ka2-ready*`.",
        ],
        "sources": [
            "particlestream/tier1.py",
            "particlestream/transport.py",
            "particlestream/memory.py",
            "configs/forge/ideas/online-ka2.json",
            "configs/forge/ideas/online-ka2-ready.json",
            "configs/forge/ideas/online-ka2-transport-causal.json",
        ],
        "references": [
            {
                "title": "Adam: A Method for Stochastic Optimization",
                "url": "https://arxiv.org/abs/1412.6980",
                "component": "Second-moment statistics $v_t$ and bias-correction readiness $1-\\beta_2^t$ used by `KA2CriticAdam`.",
            },
            {
                "title": "Which Training Methods for GANs do actually Converge?",
                "url": "https://arxiv.org/abs/1801.04406",
                "component": "Zero-centered real-input gradient regularization motivating the early RMS penalty $A$.",
            },
            {
                "title": "The relativistic discriminator: a key element missing from standard GAN",
                "url": "https://arxiv.org/abs/1807.00734",
                "component": "Paired relativistic logistic loss used in the default `ka2` preset.",
            },
        ],
    },
    "FastWeight-BCAP": {
        "overview": (
            "FastWeight-BCAP extends the preset `bcap_adam` trainer with a detached Hebbian fast-weight "
            "memory module (`FastWeightMemory`, $d_c=16$) that updates only on real arrivals and conditions "
            "both generator and critic without backpropagating through temporal history. Its strict online "
            "declaration (`online-fast-weight-bcap`) preserves the complete `bcap_adam` relativistic + "
            "VICReg optimization recipe while disabling frame replay."
        ),
        "symbols": (
            "$G$ is the generator, $D$ the critic, and $P$ the trainable $K$-particle MoG prior ($K=32$, "
            "$d_z=8$, $\\sigma=0.025$). At step $t$, $x_t\\in[0,1]^{1\\times 8\\times 8}$ is the arriving "
            "real frame with flattened vector $\\bar x_t=\\operatorname{vec}(x_t)\\in\\mathbb R^{64}$. "
            "$W_k,W_v\\in\\mathbb R^{d_c\\times 64}$ ($d_c=16$) are fixed projections inside `FastWeightMemory`, "
            "$M_t\\in\\mathbb R^{d_c\\times d_c}$ is the detached Hebbian fast-weight matrix with decay "
            "$\\gamma=0.90$ and write rate $\\eta_w=0.35$, and $c_t\\in\\mathbb R^{d_c}$ is the detached "
            "velocity context passed to step $t+1$. $\\Delta(x,y)=D(x,c_{t-1})-D(y,c_{t-1})$ is the paired "
            "relativistic score difference and $R_{\\mathrm{VICReg}}(P)$ is the preset prior regularizer."
        ),
        "equations": [
            {
                "label": "Detached Hebbian Writer and Thresholded Velocity Reader",
                "latex": (
                    "\\begin{aligned}"
                    "k_t&=\\frac{W_k\\bar x_t}{\\lVert W_k\\bar x_t\\rVert_2},\\qquad "
                    "v_t=\\frac{W_v\\bar x_t}{\\lVert W_v\\bar x_t\\rVert_2},\\qquad "
                    "M_t=\\gamma M_{t-1}+\\eta_w\\,v_t k_{t-1}^\\top,\\\\"
                    "c_t&=\\begin{cases}3.0\\,\\dfrac{k_t-k_{t-1}}{\\lVert k_t-k_{t-1}\\rVert_2},&"
                    "\\lVert k_{t-1}\\rVert_2>0\\land\\lVert k_t-k_{t-1}\\rVert_2>0.35,\\\\[6pt]"
                    "\\mathbf 0_{d_c},&\\text{otherwise}.\\end{cases}"
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Only real observations mutate $(M_t,k_{t-1})$ inside `torch.no_grad()`. Because the "
                    "original `velocity` reader emits the normalized key difference $k_t-k_{t-1}$ without "
                    "querying $M_t$, it cleanly disambiguates 1-step motion (`translating_bar_next`) but "
                    "collapses to $\\mathbf 0_{d_c}$ across repeated distractors ($k_t=k_{t-1}$)."
                ),
            },
            {
                "label": "Single-Arrival Relativistic BCAP and VICReg Objectives",
                "latex": (
                    "\\begin{aligned}"
                    "\\mathcal L_D&=\\operatorname{softplus}\\big(-\\Delta(x_t,\\hat x_D)\\big)+"
                    "\\frac{\\lambda}{2}\\Big[(\\lVert g_r\\rVert_2-\\kappa)_+^2+(\\lVert g_f\\rVert_2-\\kappa)_+^2\\Big],\\\\"
                    "\\mathcal L_G&=\\operatorname{softplus}\\big(\\Delta(x_t,\\hat x_G)\\big)+"
                    "\\lambda_{\\mathrm{prior}}R_{\\mathrm{VICReg}}(P)."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "With empty `recipe_overrides`, `online-fast-weight-bcap` inherits `bcap_adam`'s "
                    "constant-LR relativistic pairing and repulsive `vicreg` prior regularizer over uniform "
                    "`batch_size=1` draws $\\hat x_D,\\hat x_G$."
                ),
            },
        ],
        "pseudocode": [
            "Initialize G, D, trainable MoG prior P and detached FastWeightMemory (d_c = 16).",
            "For each chronological single-arrival step t:",
            "  On predictive streams at arrival 0, warm-start c_0 = FastWeightMemory.write(x_0) without stepping G or D.",
            "  Serve pre-arrival prediction y_hat = G(z_{i_serve}, c_{t-1}) under isolated eval RNG.",
            "  Draw uniform i_D and compute detached fake x_hat_D = stop_gradient(G(z_{i_D}, c_{t-1})).",
            "  Backpropagate relativistic critic loss softplus(D(x_hat_D, c_{t-1}) - D(x_t, c_{t-1})) + BCAP cap; step D.",
            "  With D fixed, draw uniform i_G, compute x_hat_G = G(z_{i_G}, c_{t-1}), and evaluate relativistic G loss + VICReg prior loss.",
            "  Step G and trainable prior P at constant preset learning rates.",
            "  Update detached M_t and emit thresholded velocity context c_t = FastWeightMemory.write(x_t) (zero replay).",
        ],
        "training_details": {
            "loss": "Preset paired relativistic logistic loss (`loss=\"relativistic\"`) on raw critic scores.",
            "optimizer": "Native Adam (`bcap_adam` preset: $\\beta_1=0$, $\\beta_2=0.999$).",
            "learning_rates": "Constant unannealed `bcap_adam` rates (`lr=0.00425`, `critic_lr_mult=1.0`, `prior_lr_mult=2.0`, `lr_floor=1.0`).",
            "gradient_clipping": "None (`critic_grad_guard=False`).",
            "critic_penalty": "Fixed real/fake one-sided squared $L_2$ BCAP cap ($\\lambda=1.0$, $\\kappa=1.0$).",
            "damping_and_guards": "None (`latent_damping_max_rate=0.0`); `prior_regularizer=\"vicreg\"` spreads trainable latent centers.",
            "online_memory_and_routing": "Detached `FastWeightMemory` (`dim=16`, `decay=0.9`, `write_lr=0.35`, `reader=\"velocity\"`), uniform particle sampling, `replay_policy=\"none\"`, `rehearsal_batch=0`.",
            "noise": "Zero scheduled critic-input or generator-output additive noise; task-declared MoG $\\sigma=0.025$ is preserved.",
            "averaging": "Live unaveraged weights (`ema_decay=0.0`, `serve_average=False`).",
        },
        "configuration_notes": [
            "`online-fast-weight-bcap` passes `seq_img_translating_bar_next` (`1/10` required passes), proving that detached key-difference conditioning alone resolves 1-step interior motion ambiguity without replay.",
            "On unconditional multi-mode streams (`bars4`, `dwell_bars4`), pairing `batch_size=1` relativistic score subtraction with unannealed updates and repulsive `vicreg` prior regularization prevents stable 4-mode convergence.",
            "The historical parent `fast-weight-bcap` omitted explicit `replay_policy=\"none\"` fields and remains preserved as read-only provenance.",
        ],
        "sources": [
            "particlestream/models.py",
            "particlestream/memory.py",
            "particlestream/online.py",
            "particlestream/tier1.py",
            "configs/forge/ideas/online-fast-weight-bcap.json",
        ],
        "references": [
            {
                "title": "Using Fast Weights to Attend to the Recent Past",
                "url": "https://arxiv.org/abs/1610.06258",
                "component": "Detached fast-weight associative state updated online from arriving activations.",
            },
            {
                "title": "The relativistic discriminator: a key element missing from standard GAN",
                "url": "https://arxiv.org/abs/1807.00734",
                "component": "Paired relativistic logistic objective inherited from the `bcap_adam` preset.",
            },
            {
                "title": "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning",
                "url": "https://arxiv.org/abs/2105.04906",
                "component": "Variance-covariance prior regularizer inherited from the `bcap_adam` preset.",
            },
        ],
    },
    "Global-Stream-BCAP": {
        "overview": (
            "Global-Stream-BCAP combines `FastWeightMemory` ($d_c=16$ velocity reader) and the fixed BCAP "
            "critic cap with one global `batch_size=1` optimization recipe: decoupled non-saturating "
            "logistic loss, an unregularized trainable prior (`prior_regularizer=\"none\"`), faster Adam "
            "second-moment tracking ($\\beta_2=0.99$), and cosine learning-rate annealing. Its strict "
            "online declaration (`online-fast-weight-bcap-ns`) removes the 7-frame reservoir rehearsal "
            "batch used by the historical `fast-weight-bcap-ns` recipe."
        ),
        "symbols": (
            "$G$ is the generator, $D$ the critic, and $P$ the trainable $K$-particle MoG prior ($K=32$, "
            "$d_z=8$, $\\sigma=0.025$) with centers $\\{\\mu_i\\}_{i=0}^{K-1}$. $x_t$ is the single real "
            "arrival at step $t\\in\\{1,\\dots,T\\}$, $c_{t-1}\\in\\mathbb R^{16}$ is the detached velocity "
            "context from `FastWeightMemory`, $\\hat x_D=G(\\mu_{i_D}+\\sigma\\epsilon_D,c_{t-1})$ is the "
            "detached critic fake, and $\\hat x_G=G(\\mu_{i_G}+\\sigma\\epsilon_G,c_{t-1})$ is the "
            "differentiable generator fake. $\\eta_0=0.005$ is the base learning rate, $u_t=t/T$ is "
            "normalized progress, $u_0=0.5$ is the cosine anneal start, and $f=0.05$ is the LR floor."
        ),
        "equations": [
            {
                "label": "Decoupled Non-Saturating BCAP Objective with Unregularized Prior",
                "latex": (
                    "\\begin{aligned}"
                    "\\mathcal L_D&=\\operatorname{softplus}(-D(x_t,c_{t-1}))+\\operatorname{softplus}(D(\\hat x_D,c_{t-1}))+"
                    "\\frac{\\lambda}{2}\\Big[(\\lVert g_r\\rVert_2-\\kappa)_+^2+(\\lVert g_f\\rVert_2-\\kappa)_+^2\\Big],\\\\"
                    "\\mathcal L_G&=\\operatorname{softplus}(-D(\\hat x_G,c_{t-1}))."
                    "\\end{aligned}"
                ),
                "explanation": (
                    "Replacing relativistic pairing with independent real/fake logistic terms avoids "
                    "single-arrival gradient inversion when $x_t$ and $\\hat x_G$ belong to different valid "
                    "modes, while `prior_regularizer=\"none\"` lets the 32 MoG centers cluster onto discrete "
                    "templates without repulsive spreading."
                ),
            },
            {
                "label": "Delayed Cosine Learning-Rate Schedule",
                "latex": (
                    "\\eta(t)=\\begin{cases}"
                    "\\eta_0,&u_t\\le u_0,\\\\[6pt]"
                    "\\eta_0\\left(f+\\dfrac{1-f}{2}\\left[1+\\cos\\!\\left(\\pi\\dfrac{u_t-u_0}{1-u_0}\\right)\\right]\\right),&u_t>u_0."
                    "\\end{cases}"
                ),
                "explanation": (
                    "Network groups step at $\\eta(t)$ and trainable prior centers step at $2\\eta(t)$ "
                    "(`prior_lr_mult=2.0`), holding full plasticity through the first half of the stream "
                    "before cooling to $5\\%$ of $\\eta_0$."
                ),
            },
        ],
        "pseudocode": [
            "Initialize G, D, unregularized trainable MoG prior P and detached FastWeightMemory (d_c = 16).",
            "For each chronological single-arrival step t of T:",
            "  Update network LR eta(t) and prior LR 2 * eta(t) from the delayed cosine schedule (u_0 = 0.5, floor = 0.05).",
            "  Serve pre-arrival prediction y_hat = G(z_{i_serve}, c_{t-1}) under isolated eval RNG when predictive.",
            "  Draw uniform i_D and compute detached fake x_hat_D = stop_gradient(G(z_{i_D}, c_{t-1})).",
            "  Backpropagate non-saturating critic loss + fixed BCAP cap (lambda = 1, kappa = 1) on x_t alone; step D.",
            "  With D fixed, draw uniform i_G, compute x_hat_G = G(z_{i_G}, c_{t-1}), and backpropagate softplus(-D(x_hat_G, c_{t-1})).",
            "  Step G and trainable prior row mu_{i_G} without prior repulsion or generator probes.",
            "  Advance detached velocity context c_t = FastWeightMemory.write(x_t) with rehearsal_batch = 0.",
        ],
        "training_details": {
            "loss": "Decoupled non-saturating logistic loss (`loss=\"non_saturating\"`) without auxiliary reconstruction.",
            "optimizer": "Native Adam (`bcap_adam`: $\\beta_1=0$, $\\beta_2=0.99$, `prior_betas=(0, 0.99)`).",
            "learning_rates": "Base LR $0.005$, `critic_lr_mult=1.0`, `prior_lr_mult=2.0`, holding until `lr_anneal_start=0.5` then cosine-decaying to `lr_floor=0.05` (`prior_lr_floor=0.05`).",
            "gradient_clipping": "None (`critic_grad_guard=False`).",
            "critic_penalty": "Fixed real/fake one-sided squared $L_2$ BCAP cap ($\\lambda=1.0$, $\\kappa=1.0$).",
            "damping_and_guards": "None (`latent_damping_max_rate=0.0`, `prior_regularizer=\"none\"`).",
            "online_memory_and_routing": "Detached `FastWeightMemory` (`dim=16`, `reader=\"velocity\"`), uniform unrouted particle draws. `online-fast-weight-bcap-ns` enforces `replay_policy=\"none\"` and `rehearsal_batch=0` (removing historical `fast-weight-bcap-ns`'s `rehearsal_batch=7`).",
            "noise": "Zero scheduled critic-input or generator-output additive noise (`0.0`); task MoG $\\sigma=0.025$ is preserved.",
            "averaging": "Live unaveraged weights (`ema_decay=0.0`, `serve_average=False`).",
        },
        "configuration_notes": [
            "Global-Stream-BCAP isolates the exact boundary between replay-assisted streaming and true single-stream online learning: removing the 7-frame FIFO rehearsal batch drops `fast-weight-bcap-ns` to `0/10` required passes (`online-fast-weight-bcap-ns`).",
            "Without rehearsal, `dwell=16` identical-frame blocks overwrite the critic and pull unrouted trainable particles toward whichever mode currently occupies the stream (`seq_img_dwell_bars4` collapses to 2/4 modes), motivating the subsequent causal-state, context-balanced, and observed-feature prior families.",
        ],
        "sources": [
            "particlestream/models.py",
            "particlestream/memory.py",
            "particlestream/online.py",
            "particlestream/tier1.py",
            "configs/forge/ideas/online-fast-weight-bcap-ns.json",
            "configs/forge/ideas/fast-weight-bcap-ns.json",
        ],
        "references": [
            {
                "title": "Generative Adversarial Networks",
                "url": "https://arxiv.org/abs/1406.2661",
                "component": "Non-saturating logistic adversarial objective replacing relativistic score pairing.",
            },
            {
                "title": "SGDR: Stochastic Gradient Descent with Warm Restarts",
                "url": "https://arxiv.org/abs/1608.03983",
                "component": "Cosine learning-rate annealing schedule applied over the second half of the stream horizon.",
            },
            {
                "title": "Using Fast Weights to Attend to the Recent Past",
                "url": "https://arxiv.org/abs/1610.06258",
                "component": "Detached fast-weight memory state maintained without backpropagating through time.",
            },
        ],
    },
}


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def validate_family_documentation(root: Path | None = None) -> dict[str, dict]:
    """Validate descriptive family metadata without touching grading or receipts."""
    required = {
        "overview",
        "symbols",
        "equations",
        "pseudocode",
        "training_details",
        "configuration_notes",
        "sources",
        "references",
    }
    detail_keys = {key for key, _ in TRAINING_DETAILS}
    for family, doc in FAMILY_DOCUMENTATION.items():
        if not isinstance(doc, dict) or set(doc) != required:
            raise ValueError(f"family documentation for {family} must have keys {sorted(required)}")
        _nonempty_text(doc["overview"], f"{family} overview")
        _nonempty_text(doc["symbols"], f"{family} symbols")
        if "](" in doc["symbols"]:
            raise ValueError(f"{family} symbols must not contain Markdown links")
        equations = doc["equations"]
        if not isinstance(equations, list) or not equations:
            raise ValueError(f"{family} equations must be a nonempty list")
        for eq in equations:
            if not isinstance(eq, dict) or set(eq) != {"label", "latex", "explanation"}:
                raise ValueError(f"{family} equation needs label, latex and explanation")
            for key in ("label", "latex", "explanation"):
                _nonempty_text(eq[key], f"{family} equation {key}")
            if any(token in eq["latex"] for token in ("$", "```", "~~~")) or re.search(r"(?<!\\)\\[\[\]()]", eq["latex"]):
                raise ValueError(f"{family} equation LaTeX must omit math delimiters and Markdown fences")
            if "](" in eq["latex"] or "](" in eq["explanation"]:
                raise ValueError(f"{family} equation text must not contain accidental Markdown links")
        pseudocode = doc["pseudocode"]
        if not isinstance(pseudocode, list) or not pseudocode:
            raise ValueError(f"{family} pseudocode must be a nonempty list")
        for line in pseudocode:
            _nonempty_text(line, f"{family} pseudocode line")
            if "```" in line or "](" in line:
                raise ValueError(f"{family} pseudocode cannot contain code fences or Markdown links")
        details = doc["training_details"]
        if not isinstance(details, dict) or set(details) != detail_keys:
            raise ValueError(f"{family} training_details must define {sorted(detail_keys)}")
        for key in detail_keys:
            _nonempty_text(details[key], f"{family} training_details[{key}]")
        notes = doc["configuration_notes"]
        if not isinstance(notes, list) or not notes:
            raise ValueError(f"{family} configuration_notes must be a nonempty list")
        for note in notes:
            _nonempty_text(note, f"{family} configuration note")
        sources = doc["sources"]
        if not isinstance(sources, list) or not sources or len(sources) != len(set(sources)):
            raise ValueError(f"{family} sources must be a nonempty list of distinct paths")
        for src in sources:
            _nonempty_text(src, f"{family} source")
            path = Path(src)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{family} source must be a relative repository path: {src}")
            if root is not None and not (Path(root) / path).is_file():
                raise ValueError(f"{family} source does not exist under {root}: {src}")
        references = doc["references"]
        if not isinstance(references, list):
            raise ValueError(f"{family} references must be a list")
        urls = []
        for ref in references:
            if not isinstance(ref, dict) or set(ref) != {"title", "url", "component"}:
                raise ValueError(f"{family} reference needs title, url and component")
            for key in ("title", "url", "component"):
                _nonempty_text(ref[key], f"{family} reference {key}")
            parsed = urlsplit(ref["url"])
            if parsed.scheme != "https" or not parsed.netloc or any(ch.isspace() for ch in ref["url"]):
                raise ValueError(f"{family} reference URL must be HTTPS: {ref['url']}")
            urls.append(ref["url"])
        if len(urls) != len(set(urls)):
            raise ValueError(f"{family} references must not repeat URLs")
    return deepcopy(FAMILY_DOCUMENTATION)
