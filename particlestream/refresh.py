"""Complete, sequential recipe measurements without changing qualification policy."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
import platform

from particlegan_forge import atomic_json, file_hash, stable_hash

from .online import validate_execution_policy


def now():
    return datetime.now(timezone.utc).isoformat()


def execution_session(device, cohort):
    """Record execution context without adding hardware to recipe ownership."""
    import torch

    session = {"started_at": now(), "requested_device": device,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "python_version": platform.python_version(),
               "torch_version": str(torch.__version__),
               "particlegan_version": version("particlegan")}
    if cohort == "cuda":
        selected_device = "cuda" if device == "auto" else device
        session["device_name"] = torch.cuda.get_device_name(selected_device)
    else:
        session["device_name"] = "cpu"
    return session


def validate_controls(tasks):
    """Exercise each task's production scorer with oracle and broken outputs."""
    import torch
    from .minimal_adapter import validate_controls as minimal_controls
    from .robotics_adapter import validate_controls as robotics_controls
    from .streams import (bars4_templates, translating_bar_cycle, fork_templates,
                          score_multimodal_bars, score_translating_next, score_fork_rollout)
    from .challenge_adapter import make_challenge_stream
    from .challenges import (score_rare_mode_bars, score_delayed_cue,
                             score_persistent_fork, score_circle_orbit, score_square_circle_cycle)

    results = {}
    for task in tasks:
        if task.adapter != "seq_image":
            continue
        mode = task.host_definition["mode"]
        if mode == "bars4":
            bank = bars4_templates()
            good = score_multimodal_bars(bank.repeat(8, 1, 1, 1), bank)
            bad = score_multimodal_bars(bank[:1].repeat(32, 1, 1, 1), bank)
        elif mode == "translating_next":
            bank, _ = translating_bar_cycle()
            right, left = bank[2:3].repeat(16, 1, 1, 1), bank[:1].repeat(16, 1, 1, 1)
            good = score_translating_next(right, left, bank)
            bad = score_translating_next(left, right, bank)
        elif mode == "fork_rollout":
            bank = fork_templates()
            good = score_fork_rollout(bank[1:3].repeat(16, 1, 1, 1), bank)
            bad = score_fork_rollout(bank[1:2].repeat(32, 1, 1, 1), bank)
        else:
            stream = make_challenge_stream(mode, "cpu")
            if mode == "variable_dwell":
                bank = stream.templates
                good = score_multimodal_bars(bank.repeat(8, 1, 1, 1), bank)
                bad = score_multimodal_bars(bank[:1].repeat(32, 1, 1, 1), bank)
            elif mode == "rare_mode":
                bank = stream.templates
                oracle = torch.cat([bank[i:i + 1].repeat(n, 1, 1, 1) for i, n in enumerate((10, 10, 10, 2))])
                missing = oracle.clone()
                missing[-2:] = bank[:1]
                good, bad = score_rare_mode_bars(oracle, bank), score_rare_mode_bars(missing, bank)
            elif mode == "delayed_cue":
                left, right = [stream.outcomes[i:i + 1].repeat(32, 1, 1, 1) for i in (0, 1)]
                good = score_delayed_cue(left, right, stream.outcomes)
                bad = score_delayed_cue(left, left, stream.outcomes)
            elif mode == "persistent_fork":
                oracle = stream.branch_paths.repeat(16, 1, 1, 1, 1)
                switching = oracle.clone()
                switching[:, 1] = oracle.flip(0)[:, 1]
                good = score_persistent_fork(oracle, stream.branch_paths)
                bad = score_persistent_fork(switching, stream.branch_paths)
            elif mode == "circle_orbit":
                oracle = stream.templates.unsqueeze(0).repeat(32, 1, 1, 1, 1)
                broken = oracle.clone()
                broken[:, -1] = stream.templates[0]
                good = score_circle_orbit(oracle, stream.templates)
                bad = score_circle_orbit(broken, stream.templates)
            elif mode == "square_circle_cycle":
                circle = stream.circle_templates.unsqueeze(0).repeat(32, 1, 1, 1, 1)
                square = stream.square_templates.unsqueeze(0).repeat(32, 1, 1, 1, 1)
                good = score_square_circle_cycle(circle, square, stream.square_templates, stream.circle_templates)
                bad = score_square_circle_cycle(square, circle, stream.square_templates, stream.circle_templates)
            else:
                raise ValueError(f"missing controls for {task.id}: {mode}")
        outcomes = {"oracle": good, "destructive": bad}
        results[task.id] = {}
        for name, metrics in outcomes.items():
            passed = all(b.check(metrics.get(b.metric, float("nan"))) for b in task.thresholds)
            if passed != (name == "oracle"):
                raise ValueError(f"{task.id}: unexpected {name} control result")
            results[task.id][name] = {"verdict": "PASS" if passed else "FAIL", "metrics": metrics}
    results.update(minimal_controls([t for t in tasks if t.adapter == "minimal_probe"]))
    results.update(robotics_controls([t for t in tasks if t.adapter == "gym_robotics"]))
    if set(results) != {t.id for t in tasks}:
        raise ValueError("not every task has validated controls")
    return results


def refresh_all(workspace, view_id, *, device="cuda", candidates=None, resume=False):
    """Run every selected recipe/task, continuing on numerical FAIL only.

    Stretch measurements do not grant qualification credit or bypass lower-tier
    gates. Resume accepts only current, unchanged receipts from this refresh.
    """
    from particlegan_forge.tiers import resolve_device

    workspace.validate()
    cohort = resolve_device(device)
    declarations = workspace.load_candidates()
    selected, historical = [], []
    for cid in sorted(set(candidates if candidates is not None else declarations)):
        candidate = declarations[cid]
        try:
            validate_execution_policy(candidate)
        except ValueError:
            if candidates is not None:
                raise
            historical.append(cid)
        else:
            selected.append(cid)
    if not selected:
        raise ValueError("no executable recipes selected")
    view = workspace.load_views()[view_id]
    tasks = workspace.load_tasks()
    assignments = sorted(view.assignments, key=lambda a: (a.qualification_tier, a.order))
    task_ids = list(dict.fromkeys(a.task for a in assignments))
    contract = {"view": view.to_dict(), "device": cohort, "seed": 0,
                "candidates": {cid: declarations[cid].to_dict() for cid in selected},
                "tasks": {tid: tasks[tid].to_dict() for tid in task_ids}}
    audit_path = workspace.reports_dir / "refresh-audit.json"
    if resume:
        audit = json.loads(audit_path.read_text())
        if audit["contract_sha256"] != stable_hash(contract):
            raise ValueError("refresh declarations differ; start a new refresh")
    else:
        audit = {"started_at": now(), "completed_at": None, "device": cohort,
                 "seed": 0, "contract_sha256": stable_hash(contract),
                 "candidates": selected, "tasks": task_ids, "historical_read_only": historical,
                 "planned_runs": len(selected) * len(task_ids), "runs": {}}
    audit["controls"] = validate_controls([tasks[tid] for tid in task_ids])
    audit["completed_at"] = None
    audit.setdefault("sessions", []).append(execution_session(device, cohort))
    atomic_json(audit_path, audit)
    for cid in selected:
        qualification = workspace.qualify(cid, view_id, device=device)
        current = {tid: outcome for tier in qualification["tiers"].values()
                   for tid, outcome in tier["tasks"].items()}
        for tid in task_ids:
            key = f"{cid}/{tid}"
            previous = audit["runs"].get(key)
            receipt_path = workspace.reports_dir / f"receipts/{cohort}/{cid}/{tid}.json"
            if previous and current[tid]["status"] in {"PASS", "FAIL"}:
                gif = workspace.reports_dir / previous["goal_gif"]
                if (receipt_path.is_file() and file_hash(receipt_path) == previous["receipt_sha256"]
                        and gif.is_file() and file_hash(gif) == previous["gif_sha256"]):
                    continue
            if previous:
                # A failed replacement must not leave an obsolete successful
                # completion counted in the refresh progress.
                del audit["runs"][key]
                atomic_json(audit_path, audit)
            receipt = workspace.run_task(cid, tid, device=device, seed=0)
            if receipt["verdict"] in {"INVALID", "INCOMPLETE"}:
                workspace.compile_reports(view_id=view_id, device=device)
                raise RuntimeError(f"{key}: {receipt['verdict']}: {receipt['error']}; fix execution then resume")
            artifacts = receipt["artifacts"]
            gif = workspace.reports_dir / artifacts["goal_gif"]
            if not gif.is_file() or file_hash(gif) != artifacts["gif_sha256"]:
                raise RuntimeError(f"{key}: missing or changed training GIF")
            audit["runs"][key] = {"verdict": receipt["verdict"], "completed_at": receipt["completed_at"],
                "completed_steps": receipt["completed_steps"], "wall_seconds": receipt["cost"]["wall_seconds"],
                "receipt_sha256": file_hash(receipt_path), "goal_gif": artifacts["goal_gif"],
                "gif_sha256": artifacts["gif_sha256"]}
            atomic_json(audit_path, audit)
            print(f"{len(audit['runs'])}/{audit['planned_runs']} {key} {receipt['verdict']} "
                  f"({receipt['cost']['wall_seconds']:.1f}s)", flush=True)
        workspace.compile_reports(view_id=view_id, device=device)
    audit["completed_at"] = now()
    atomic_json(audit_path, audit)
    workspace.compile_reports(view_id=view_id, device=device)
    return {"completed_runs": len(audit["runs"]), "audit": str(audit_path)}
