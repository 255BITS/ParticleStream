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
            "are context keys, visit counts, and outcome masses ($C_{\\max}=64$, $\\kappa_{\\mathrm{visit}}=0.25$, $h_k=0.12$)."
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
                    "the invariant while down-weighting nonlinear carry/parity directions ($\\rho_v=0.10$, $\\sigma_{\\mathrm{slab}}=1.0$). "
                    "On continuous manifolds ($q_t/\\operatorname{tr}(S_t)\\gg\\rho_v$), $\\beta_t(c)\\to 0$ preserves smooth "
                    "extrapolation; at recurrent multimodal contexts ($V_{b^*}>1$), empirical-Bayes shrinkage "
                    "$\\gamma_t(c)=\\frac{V_{b^*}-1}{V_{b^*}-1+\\kappa_{\\mathrm{visit}}}\\frac{\\operatorname{tr}(S_t)}{\\operatorname{tr}(S_t)+q_t+10^{-5}}\\exp(-\\lVert c-k_{b^*}^{\\mathrm{ctx}}\\rVert_2^2/h_k^2)$ "
                    "recovers exact branch quantiles ($z_i^{\\mathrm{cond}}(k_{b^*}^{\\mathrm{ctx}})\\to\\hat\\mu_{q(i,b^*)}+\\epsilon_i$)."
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
            "online_memory_and_routing": "Detached `CausalStateMemory` ($d_c=16$, `association_weight=0.0`) plus `ResidualFeatureConditionalReconstruction` (`rls_ridge=0.1`, `covariance_ema=0.05`, `slab_scale=1.0`, `variance_ratio_scale=0.10`, `visit_shrinkage=0.25`, `kernel_bandwidth=0.12`, `replay_policy=\"none\"`, `rehearsal_batch=0`).",
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
            "FeaturePrior-BCAP established the first Tier 1-qualifying global recipe (`online-feature-prior-bce`: 10/10 required passes, 1/10 stretch passes on `single_stream_stability` view 10), serving as the direct parent of ResidualPrior-BCAP.",
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
