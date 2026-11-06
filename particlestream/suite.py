"""Current suite policy, keeping the original adapter source bytes intact."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from particlegan_forge import TierSpec, TierView, atomic_text

from .forge import StreamWorkspace
from .tier1 import CANDIDATES, IMPLEMENTATION, REGISTRY, TIER1_TASKS
from .minimal_adapter import make_tasks, run_minimal_task
from .robotics_adapter import make_tasks as make_robotics_tasks, run_robotics_task

MINIMAL_TASKS = make_tasks(IMPLEMENTATION)
ROBOTICS_TASKS = make_robotics_tasks(IMPLEMENTATION)
ALL_TASKS = [*TIER1_TASKS, *MINIMAL_TASKS, *ROBOTICS_TASKS]
REGISTRY.adapter("minimal_probe")(run_minimal_task)
REGISTRY.adapter("gym_robotics")(run_robotics_task)
PRIMARY_VIEW = TierView.from_tiers(
    id="single_stream_stability", goal="single_stream_stability", revision=10,
    policy_change_reason="Replace the robotics tasks' capped affine target encoding with one shared invertible arctangent map after a fast Slide transition exceeded the range; preserve all recipes, physical gates, budgets, required tasks and minimal probes.",
    tiers=[TierSpec(tier=tier,
                    required=[task.id for task in TIER1_TASKS] if tier == 1 else [],
                    stretch=[task.id for task in [*MINIMAL_TASKS, *ROBOTICS_TASKS]
                             if task.host_definition["qualification_tier"] == tier])
           for tier in (1, 2, 3)],
)


def bootstrap_workspace(root: Path | str) -> StreamWorkspace:
    workspace = StreamWorkspace(root, registry=REGISTRY)
    view_path = workspace.views_dir / f"{PRIMARY_VIEW.id}.json"
    if view_path.is_file():
        previous = view_path.read_text()
        revision = json.loads(previous)["revision"]
        if revision > PRIMARY_VIEW.revision:
            raise ValueError("bootstrap would replace a newer view; reconcile its policy first")
        if revision < PRIMARY_VIEW.revision:
            archive = workspace.view_history_dir / f"{PRIMARY_VIEW.id}-v{revision}.json"
            if archive.exists() and archive.read_text() != previous:
                raise ValueError("archived view revision differs; refusing to overwrite history")
            atomic_text(archive, previous)
    for task in ALL_TASKS:
        workspace.register_task(task)
    workspace.register_view(PRIMARY_VIEW)
    for candidate in CANDIDATES:
        workspace.register_candidate(candidate)
    workspace.compile_reports(view_id=PRIMARY_VIEW.id, device="cuda" if torch.cuda.is_available() else "cpu")
    return workspace
