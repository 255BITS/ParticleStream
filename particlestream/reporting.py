"""Linked technique explanations and evidence, derived from Forge qualification.

This presentation layer never regrades, rebinds or selects outcomes per task.
The inventory is the only ranked leaderboard; family pages explain exact recipes.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from particlegan_forge import atomic_text
from particlegan_forge.tiers import resolve_device

from .family_documentation import FAMILY_DOCUMENTATION, TRAINING_DETAILS, validate_family_documentation
from .online import validate_execution_policy


METHODS = {
    "BCAP": (
        "BCAP penalizes critic input-gradient norms above a fixed cap on real and generated examples. "
        "The cap is a soft term in the critic loss. The generator and trainable prior still learn through "
        "the adversarial objective. Adam versus the preset optimizer, learning rates, annealing, prior "
        "updates and reconstruction are recipe choices, shown separately below.",
        "tier1.py",
    ),
    "BISPF": (
        "Born-Infeld Sinkhorn Particle Flow combines an elastic critic-gradient penalty on real, fake "
        "and transport-chord inputs with a zero-mean logit gauge. Its public presets also provide "
        "particle-prior regularization and thermal policies. ParticleStream's declared recipes override "
        "some preset choices: inspect each exact configuration rather than assuming every BISPF option "
        "is enabled. With a real and fake batch of one, a Sinkhorn coupling has only one pair.",
        "tier1.py",
    ),
    "KA2": (
        "KA2 uses a critic-local controller: input-gradient penalties and an averaged critic-gradient "
        "anchor are governed by penalty-call history and optimizer moment surprise. Noise, averaging "
        "for serving, readiness and damping remain configuration choices. In this short online suite, "
        "the ready recipes explicitly change critic beta2 so moment readiness can engage within "
        "480 updates; that does not change a task's budget or gates.",
        "tier1.py",
    ),
    "FastWeight-BCAP": (
        "FastWeight-BCAP adds detached bounded memory to the BCAP trainer. The memory writes from "
        "arriving frames and conditions the task-owned generator and critic. Its strict online "
        "descendant disables replay. The original reader emphasizes recent frame change; it does not "
        "by itself establish delayed recall or use the accumulated associative matrix as a query.",
        "memory.py",
    ),
    "Global-Stream-BCAP": (
        "Global-Stream-BCAP combines BCAP and bounded frame memory with one declared non-saturating "
        "loss and cosine learning-rate schedule across tasks. Its strict online descendant retains "
        "these optimizer settings while removing rehearsal. The historical recipe's rehearsal buffer "
        "must not be confused with the current one-arrival, one-update execution contract.",
        "memory.py",
    ),
    "FeaturePrior-BCAP": (
        "FeaturePrior-BCAP adds a learned conditional prior to BCAP and a causal memory reader. "
        "A fixed named orthogonal 8×64 projection encodes each arriving image. At most 32 feature "
        "means and 64 context/outcome summaries accumulate after successful updates. Serving draws "
        "empirical feature quantiles near the observed context while retaining every row's MoG "
        "perturbation. The storage prior is frozen; a weight-10 BCE current-observation posterior "
        "trains the image decoder. Serving receives neither the current target nor stored images, "
        "and this policy uses no generator-center probes.",
        "feature_prior.py",
    ),
    "ResidualPrior-BCAP": (
        "ResidualPrior-BCAP replaces FeaturePrior-BCAP's nearest-neighbor context lookup on unseen "
        "states with a closed-form online Sherman-Morrison RLS affine predictor, a precision-weighted "
        "Mahalanobis spike-and-slab prototype posterior, and an empirical-Bayes kernel-gated live "
        "residual quantile transform, paired with a Markov bottleneck penalty on direct generator "
        "context-shortcut weights. All statistics live in detached buffers updated strictly after "
        "successful batch_size=1 arrivals without replay, generator probes, or target retention.",
        "residual_prior.py",
    ),
    "GAIL-BCAP": (
        "GAIL-BCAP adapts stabilized generative adversarial imitation control to strict batch-one "
        "online world modeling. A Hinge critic with BCAP input-gradient capping pairs on the "
        "generator with a prioritized reconstruction anchor and a bounded sigmoid critic style "
        "bonus (canceling unbounded linear generator slopes), modulated by a causal discounted "
        "return advantage baseline and entropy-regularized categorical outcome quantiles. Unseen "
        "contexts extrapolate through a detached Sherman-Morrison associative fast-weight operator "
        "(radial distance-blended or precision-weighted Mahalanobis spike-and-slab with kernel-gated "
        "residual quantiles, Markov context-bypass regularization, precision-whitened Gram-Schmidt "
        "innovation subspaces, Matérn-3/2 kernel ridge residuals, causal Dirichlet-Sobolev and "
        "wavepacket shift-equivariant calibration, and causal primal-dual Schedule-Free "
        "generator optimization) without storing or replaying frames.",
        "gail_prior.py",
    ),
    "SubspacePrior-BCAP": (
        "SubspacePrior-BCAP replaces blind fixed 8×64 random projection on the leading six latent "
        "coordinates with a causally anchored two-pass Gram-Schmidt innovation subspace scaled once at "
        "discovery by precision-whitening gains, paired with a 2-row deterministic Johnson-Lindenstrauss "
        "tail and a semiparametric global RLS plus unquantized Nadaraya-Watson kernel-ridge residual "
        "prior. This eliminates random-projection nullspace loss and 1:1 MoG Wiener attenuation on "
        "millimetre physical micro-displacements without retroactive coordinate drift or stored frames.",
        "subspace_prior.py",
    ),
}


def slug(technique):
    return technique.lower().replace(" ", "-")


def _anchor(value):
    return f'<a id="{value}"></a>'


def _table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ")
                                  for value in row) + " |" for row in rows], ""]


def _outcomes(qualification, *, role=None, tier=None):
    return [(tid, outcome) for number, summary in qualification["tiers"].items()
            if tier is None or number == tier
            for tid, outcome in summary["tasks"].items()
            if role is None or outcome["importance"] == role]


def _counts(outcomes):
    return (sum(o["status"] == "PASS" for _, o in outcomes), len(outcomes),
            sum(o["status"] in {"PASS", "FAIL"} for _, o in outcomes))


def _recipe_link(candidate, suffix="", *, family_page=False):
    path = f"{slug(candidate.technique)}.md" if family_page else f"families/{slug(candidate.technique)}.md"
    return f"{path}#{candidate.id}{suffix}"


def _score(candidate, outcomes, suffix, *, family_page=False):
    passed, total, measured = _counts(outcomes)
    if not total:
        return "—"
    label = f"{passed}/{total}" + ("*" if measured < total else "")
    return f"[{label}]({_recipe_link(candidate, suffix, family_page=family_page)})"


def _rank(candidate_id, qualification):
    passed, total, measured = _counts(_outcomes(qualification, role="required"))
    stretch_passed, _, _ = _counts(_outcomes(qualification, role="stretch"))
    # A representative must be a complete global recipe when one is available.
    return (-(measured == total), -qualification["qualified_tier"], -passed,
            -stretch_passed, -_counts(_outcomes(qualification))[2], candidate_id)


def _json_delta(previous, current, prefix=""):
    rows = []
    for key in sorted(set(previous) | set(current)):
        path = f"{prefix}.{key}" if prefix else key
        before, after = previous.get(key), current.get(key)
        if key in previous and key in current and isinstance(before, dict) and isinstance(after, dict):
            rows.extend(_json_delta(before, after, path))
        elif key not in previous or key not in current or before != after:
            render = lambda obj, present: "_unset_" if not present else f"`{json.dumps(obj, sort_keys=True)}`"
            rows.append([f"`{path}`", render(before, key in previous), render(after, key in current)])
    return rows


def _receipt(workspace, outcome):
    path = outcome.get("receipt")
    if outcome["status"] == "NOT_RUN" or not path:
        return None
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _evidence(workspace, candidate, task, outcome, device):
    receipt = _receipt(workspace, outcome)
    links = []
    stamp = "—"
    if receipt:
        links.append(f"[receipt](../receipts/{device}/{candidate.id}/{task.id}.json)")
        stamp = receipt.get("completed_at", receipt.get("timestamp", "—"))
    if outcome.get("goal_gif"):
        links.append(f"[goal.gif](../{outcome['goal_gif']})")
    readings = "; ".join(f"`{bound.metric}={outcome.get('live', {}).get(bound.metric, '—'):.5g}`"
                         if isinstance(outcome.get("live", {}).get(bound.metric), (int, float))
                         else f"`{bound.metric}=—`" for bound in task.thresholds)
    failed = "; ".join(outcome.get("failed_bounds", []))
    reason = outcome.get("reason", "")
    if outcome["status"] == "NOT_RUN":
        readings = reason or "No current receipt"
    elif failed:
        readings += f"; failed terminal bounds: {failed}"
    elif reason and outcome["status"] not in {"PASS", "FAIL"}:
        readings += f"; {reason}"
    terminal = (f"{outcome.get('terminal_passing', 0)}/{task.minimum_stable_checks}"
                if receipt else "—")
    return [f"[`{task.id}`](../technique-inventory.md#{task.id})", f"**{outcome['status']}**",
            terminal, readings, " · ".join(links) or "—", stamp]


def _family_documentation_lines(technique, explanation, source, documentation):
    doc = documentation.get(technique)
    page = f"{slug(technique)}.md"
    if doc is None:
        return [
            _anchor("technique-overview"),
            "## Technique overview",
            "",
            explanation,
            "",
            f"[Implementation](../../../particlestream/{source}) · "
            "[Protocol](../../../EXPERIMENTATION.md) · [Terminology](../../../RESEARCH_GLOSSARY.md)",
            "",
        ]
    nav = [
        f"[Implementation](../../../particlestream/{source})",
        f"[Mathematical formulation]({page}#mathematical-formulation)",
        f"[Simplified pseudocode]({page}#simplified-pseudocode)",
        f"[Training details]({page}#training-details)",
        "[Protocol](../../../EXPERIMENTATION.md)",
        "[Terminology](../../../RESEARCH_GLOSSARY.md)",
    ]
    lines = [
        _anchor("technique-overview"),
        "## Technique overview",
        "",
        doc["overview"],
        "",
        " · ".join(nav),
        "",
        _anchor("mathematical-formulation"),
        "## Mathematical formulation",
        "",
        doc["symbols"],
        "",
    ]
    for equation in doc["equations"]:
        lines.extend([
            f"**{equation['label']}**",
            "",
            "$$",
            equation["latex"],
            "$$",
            "",
            equation["explanation"],
            "",
        ])
    lines.extend([
        _anchor("simplified-pseudocode"),
        "## Simplified pseudocode",
        "",
        "```text",
        *doc["pseudocode"],
        "```",
        "",
        _anchor("training-details"),
        "## Training details",
        "",
        *_table(
            ["Characteristic", "Behavior"],
            [[label, doc["training_details"][key]] for key, label in TRAINING_DETAILS],
        ),
        _anchor("configuration-differences"),
        "## Configuration differences",
        "",
        *[f"- {note}" for note in doc["configuration_notes"]],
        "",
        "<details>",
        "<summary>Implementation and recipe sources</summary>",
        "",
        "These links support the explanation. Recorded results below remain bound to their own executed source.",
        "",
        *[f"- [{src}](../../../{src})" for src in doc["sources"]],
        "",
        "</details>",
        "",
    ])
    return lines


def _family_report(workspace, technique, members, declarations, qualifications, tasks, view, device, documentation):
    explanation, source = METHODS.get(technique, (f"Declared {technique} training family.", "tier1.py"))
    doc = documentation.get(technique)
    lines = [
        "<!-- Generated Forge family report -->",
        "",
        f"# {technique}",
        "",
        "[← Technique leaderboard](../technique-inventory.md)",
        "",
        *_family_documentation_lines(technique, explanation, source, documentation),
        "Each section below describes one exact global recipe. Configuration and parent deltas come from "
        "the declarations; results come from Forge's current source-bound receipts. Required and "
        "stretch evidence stay separate. A final metric can meet its gate while the sustained verdict "
        "fails: every one of the last five checks must pass.",
        "",
        "The installed engine is pinned to ParticleGAN/Forge 0.9.7. Older version or replay wording "
        "in declaration history describes provenance; the effective configuration below controls execution.",
        "",
    ]
    for candidate in members:
        try:
            validate_execution_policy(candidate)
            executable = True
        except ValueError:
            executable = False
        lines.extend([_anchor(candidate.id), f"## `{candidate.id}`", "",
                      f"[Exact declaration](../../../configs/forge/ideas/{candidate.id}.json)", ""])
        parent = declarations.get(candidate.parent) if candidate.parent else None
        if parent:
            lines.extend([f"Parent: [`{parent.id}`]({_recipe_link(parent, family_page=True)}).", ""])
        own_changes = [factor for factor in candidate.changed_factors
                       if not parent or factor not in parent.changed_factors]
        if own_changes:
            lines.extend(["Declared changes:", "", *[f"- {factor}" for factor in own_changes], ""])
        effective = {"recipe_preset": candidate.recipe_preset, "recipe_overrides": candidate.recipe_overrides,
                     "memory_policy": candidate.memory_policy}
        if parent:
            previous = {"recipe_preset": parent.recipe_preset, "recipe_overrides": parent.recipe_overrides,
                        "memory_policy": parent.memory_policy}
            delta = _json_delta(previous, effective)
            lines.extend(["### Configuration delta from parent", ""])
            lines.extend(_table(["Field", "Parent", "This recipe"], delta) if delta else ["No configuration delta.", ""])
        lines.extend(["### Effective declaration", "", "```json", json.dumps(effective, indent=2, sort_keys=True), "```", ""])
        if not executable:
            successors = [c for c in declarations.values() if c.parent == candidate.id
                          and c.memory_policy.get("replay_policy") == "none"]
            lines.extend(["**Historical, read-only.** This declaration does not satisfy explicit "
                          "`replay_policy=none` with `rehearsal_batch=0`; the executor rejects it before "
                          "touching its saved evidence. It has no current qualification score.", ""])
            if successors:
                lines.extend(["Strict online successors: " + ", ".join(
                    f"[`{c.id}`]({_recipe_link(c, family_page=True)})" for c in successors) + ".", ""])
            lines.extend([f"[Preserved receipt directory](../receipts/{device}/{candidate.id}) · "
                          "[Archived cohorts](../cohorts) · [Research history](../research-notes.md)", ""])
            continue
        qualification = qualifications[candidate.id]
        passed, total, measured = _counts(_outcomes(qualification, role="required"))
        spassed, stotal, smeasured = _counts(_outcomes(qualification, role="stretch"))
        lines.extend([f"Current `{device}`, seed 0: **{qualification['status']}**, "
                      f"{passed}/{total} required passes ({measured} measured), "
                      f"{spassed}/{stotal} stretch passes ({smeasured} measured); "
                      f"qualified tier {qualification['qualified_tier']}.", ""])
        receipts = [r for _, o in _outcomes(qualification) if (r := _receipt(workspace, o))]
        if receipts:
            binding = receipts[0].get("binding", {})
            lines.extend(["Receipt identity: " + "; ".join(
                f"{name} `{binding[name]}`" for name in ("candidate_fingerprint", "engine_source_sha256")
                if name in binding) + ". Each receipt also binds its task and adapter source.", ""])
        for role in ("required", "stretch", "held", "diagnostic"):
            selected = _outcomes(qualification, role=role)
            if not selected:
                continue
            lines.extend([_anchor(f"{candidate.id}-{role}"), f"### {role.title()} evidence", ""])
            for tier in sorted(qualification["tiers"]):
                outcomes = _outcomes(qualification, role=role, tier=tier)
                if not outcomes:
                    continue
                lines.extend([_anchor(f"{candidate.id}-tier-{tier}-{role}"), f"#### Tier {tier} {role}", ""])
                rows = [_evidence(workspace, candidate, tasks[tid], outcome, device) for tid, outcome in outcomes]
                lines.extend(_table(["Task and gates", "Verdict", "Terminal checks", "Final gate readings / failure", "Evidence", "Completed UTC"], rows))
        lines.extend([f"Reproduce this exact global recipe on every task: `particlestream compare --all-tasks "
                      f"--candidate {candidate.id} --device {device}`.", ""])
    lines.extend([
        _anchor("refresh"),
        "## Refresh",
        "",
        "```sh",
        f"particlestream compile --device {device}",
        "```",
        "",
        "This page is generated alongside the leaderboard from committed source-bound receipts and declarations; "
        "editing a generated page cannot change a verdict or earn qualification.",
        "",
    ])
    if doc is not None:
        lines.extend([_anchor("references"), "## References", ""])
        if not doc["references"]:
            lines.extend([
                "This technique is repository-specific; no dedicated paper reference is declared. "
                "Its implementation and recipe sources are linked above.",
                "",
            ])
        else:
            lines.extend([f"- [{ref['title']}]({ref['url']}). {ref['component']}" for ref in doc["references"]])
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def compile_linked_reports(workspace, *, view_id=None, device=None, seed=0):
    """Write the sole inventory plus linked explanatory family pages."""
    repo_root = workspace.root if (workspace.root / "particlestream").is_dir() else None
    documentation = validate_family_documentation(repo_root)
    device = resolve_device(device)
    views = workspace.load_views()
    view = views[view_id] if view_id else next(iter(views.values()))
    declarations = workspace.load_candidates()
    tasks = workspace.load_tasks()
    online, historical = [], []
    for candidate in declarations.values():
        try:
            validate_execution_policy(candidate)
        except ValueError:
            historical.append(candidate)
        else:
            online.append(candidate)
    qualifications = {c.id: workspace.qualify(c.id, view.id, device=device, seed=seed) for c in online}
    online.sort(key=lambda c: _rank(c.id, qualifications[c.id]))
    tiers = sorted({a.qualification_tier for a in view.assignments})
    families = defaultdict(list)
    for candidate in online:
        families[candidate.technique].append(candidate)
    representatives = [members[0] for members in families.values()]
    lines = [f"# Technique leaderboard — `{view.goal}` (view {view.revision})", "",
             f"Generated from `particlegan_forge` qualification of receipts under "
             f"`reports/forge/receipts/{device}`; protocol seed {seed}, sequential real/fake batch size 1.", "",
             "Click a technique for its explanation, a recipe for its exact configuration and parent delta, "
             "or a score for gates, measurements, source-bound receipts and actual-training GIFs. "
             "One global recipe supplies every score in a row. Stretch results do not grant qualification credit.", ""]
    audit_path = workspace.reports_dir / "refresh-audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text())
        if audit.get("device") == device:
            status = "Completed" if audit.get("completed_at") else "In progress"
            count = len(audit.get("runs", {}))
            controls = audit.get("controls", {})
            oracle_passes = sum(cases.get("oracle", {}).get("verdict") == "PASS" for cases in controls.values())
            destructive_failures = sum(case.get("verdict") == "FAIL" for cases in controls.values()
                                       for name, case in cases.items() if name != "oracle")
            lines.extend([f"**Latest refresh: {status.lower()}, {count}/{audit['planned_runs']} runs recorded.** "
                          f"Started {audit['started_at']}" + (f"; completed {audit['completed_at']}" if audit.get("completed_at") else "") + ". "
                          f"Controls: {oracle_passes} oracle PASS, {destructive_failures} destructive FAIL. "
                          "[Refresh audit and artifact hashes](refresh-audit.json).", ""])
    lines.extend(["`*` means at least one current task result is missing. A displayed zero counts passes; "
                  "the measured column distinguishes an observed failure from missing or stale evidence. "
                  "Historical recipes are listed separately and receive no current score.", "",
                  "## Representative recipes", "",
                  "One complete recipe per technique, ordered by required qualification and passes, then "
                  "stretch passes. Incomplete recipes are considered only when a technique has no complete "
                  "required measurement. Ties use measured coverage and recipe name.", ""])

    def main_row(candidate):
        q = qualifications[candidate.id]
        measured = _counts(_outcomes(q))[2]
        required = _outcomes(q, role="required")
        return [f"[{candidate.technique}](families/{slug(candidate.technique)}.md)",
                f"[`{candidate.id}`]({_recipe_link(candidate)})",
                *[_score(candidate, _outcomes(q, role="required", tier=t), f"-tier-{t}-required") for t in tiers],
                _score(candidate, required, "-required"), f"{measured}/{len(_outcomes(q))}", f"`{q['status']}`"]

    headers = ["Technique", "Exact global recipe", *[f"Tier {t} required" for t in tiers], "Total required", "Measured tasks", "Status"]
    lines.extend(_table(headers, [main_row(c) for c in representatives]))
    lines.extend(["## All executable recipes", ""])
    lines.extend(_table(headers, [main_row(c) for c in online]))
    lines.extend(["## Stretch experiments", "",
                  "These measurements retain their declared gates and budgets even when a required task fails. "
                  "They do not promote a recipe past an unmet lower-tier requirement.", ""])
    rows = []
    for candidate in online:
        q = qualifications[candidate.id]
        stretch = _outcomes(q, role="stretch")
        rows.append([f"[{candidate.technique}](families/{slug(candidate.technique)}.md)",
                     f"[`{candidate.id}`]({_recipe_link(candidate)})",
                     *[_score(candidate, _outcomes(q, role="stretch", tier=t), f"-tier-{t}-stretch") for t in tiers],
                     _score(candidate, stretch, "-stretch"), f"{_counts(stretch)[2]}/{len(stretch)}"])
    lines.extend(_table(["Technique", "Recipe", *[f"Tier {t} stretch" for t in tiers], "Total stretch", "Measured stretch"], rows))
    lines.extend(["## Readout and recommendation", ""])
    if online:
        best = online[0]
        q = qualifications[best.id]
        p, total, measured = _counts(_outcomes(q, role="required"))
        sp, st, sm = _counts(_outcomes(q, role="stretch"))
        if q["eligible"]:
            lines.extend([f"Use [`{best.id}`]({_recipe_link(best)}) as the measured global reference: "
                          f"it passes {p}/{total} required tasks and {sp}/{st} stretch tasks "
                          f"({sm} stretch tasks measured). Qualification reaches tier {q['qualified_tier']}. "
                          "Inspect the failing stretch gates before claiming broader capability.", ""])
        else:
            lines.extend([f"No measured recipe currently qualifies for this view. The leading measured "
                          f"recipe is [`{best.id}`]({_recipe_link(best)}), with {p}/{total} required "
                          f"passes ({measured} measured). Complete missing evidence and address the recorded "
                          "failures without changing gates or selecting recipes per task.", ""])
    lines.extend(["## Historical declarations", "",
                  "These declarations are read-only because they lack the explicit no-replay policy "
                  "required by the current executor. Their saved artifacts remain available; stale "
                  "historical receipts are not fresh failures. Strict online successors declare the replay "
                  "change under new identities.", ""])
    rows = []
    for candidate in sorted(historical, key=lambda c: (c.technique, c.id)):
        successors = [c for c in online if c.parent == candidate.id]
        rows.append([f"[{candidate.technique}](families/{slug(candidate.technique)}.md)",
                     f"[`{candidate.id}`]({_recipe_link(candidate)})", "READ_ONLY",
                     ", ".join(f"[`{c.id}`]({_recipe_link(c)})" for c in successors) or "—"])
    lines.extend(_table(["Technique", "Historical recipe", "Execution", "Strict online successor"], rows))
    lines.extend(["## Per-Task Breakdown", "",
                  "The declared gates below apply at every terminal check. Detailed final readings, "
                  "terminal failures, timestamps and receipts are linked from each recipe's evidence.", ""])
    for assignment in sorted(view.assignments, key=lambda a: (a.qualification_tier, a.order)):
        task = tasks[assignment.task]
        lines.extend([_anchor(task.id), f"### Tier {assignment.qualification_tier}: `{task.id}` ({assignment.importance})", "",
                      task.description, "", "Gates: " + ", ".join(f"`{b.metric} {b.op} {b.bound}`" for b in task.thresholds) + ". "
                      f"Budget: {task.steps} updates; {len(task.observation_steps())} checks; "
                      f"last {task.minimum_stable_checks} must pass.", ""])
        rows = []
        for candidate in online:
            outcome = qualifications[candidate.id]["tiers"][assignment.qualification_tier]["tasks"][task.id]
            evidence = f"[details]({_recipe_link(candidate, f'-tier-{assignment.qualification_tier}-{assignment.importance}')})"
            if outcome.get("goal_gif"):
                evidence += f" · [goal.gif]({outcome['goal_gif']})"
            terminal = f"{outcome.get('terminal_passing', 0)}/{task.minimum_stable_checks}" if outcome["status"] != "NOT_RUN" else "—"
            rows.append([f"[`{candidate.id}`]({_recipe_link(candidate)})", outcome["status"], terminal, evidence])
        lines.extend(_table(["Recipe", "Verdict", "Terminal checks", "Evidence"], rows))
    lines.extend(["## Reproduction", "", "```sh", "particlestream validate",
                  f"particlestream compare --all-tasks --device {device}",
                  f"# Resume only unchanged runs from this refresh, if interrupted:",
                  f"particlestream compare --all-tasks --resume --device {device}",
                  f"# Regenerate this inventory and every linked explanation without training:",
                  f"particlestream compile --device {device}", "```", "",
                  "[Protocol](../../EXPERIMENTATION.md) · [Task contracts](EXPERIMENTS_BY_TIER.md) · "
                  "[Experiment memory](EXPERIMENT_MEMORY.md) · [Historical interpretation](research-notes.md)", ""])
    atomic_text(workspace.reports_dir / "technique-inventory.md", "\n".join(lines).rstrip() + "\n")
    family_paths = []
    for technique in sorted({c.technique for c in declarations.values()}):
        members = [c for c in online if c.technique == technique]
        members += sorted([c for c in historical if c.technique == technique], key=lambda c: c.id)
        path = workspace.reports_dir / f"families/{slug(technique)}.md"
        atomic_text(path, _family_report(
            workspace, technique, members, declarations, qualifications, tasks, view, device, documentation
        ))
        family_paths.append(str(path))
    return family_paths
