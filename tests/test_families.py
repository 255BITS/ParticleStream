"""Current-wheel integration and truthful family qualification contracts."""
import json
from pathlib import Path

import pytest

from particlestream.forge import StreamWorkspace, main
from particlestream.tier1 import (
    FAMILY_PRESETS, PRIMARY_VIEW, TIER1_CANDIDATES, TIER1_TASKS,
    _build_recipe, bootstrap_workspace,
)


@pytest.mark.parametrize("candidate_id", ("online-bcap-adam", "online-bispf-early", "online-ka2-ready"))
def test_family_runs_through_current_public_trainer(tmp_path, candidate_id):
    workspace = bootstrap_workspace(tmp_path)
    receipt = workspace.run_task(candidate_id, TIER1_TASKS[0].id,
                                 device="cpu", steps_override=4)
    family = {"online-bcap-adam": "BCAP", "online-bispf-early": "BISPF", "online-ka2-ready": "KA2"}[candidate_id]
    assert receipt["grade"]["status"] == "INCOMPLETE", receipt["error"]
    assert receipt["all_finite"] is True
    assert receipt["unintended_rng_deviations"] == 0
    assert receipt["artifacts"]["particlegan_version"] == "0.9.7"
    recipe = receipt["artifacts"]["resolved_recipe"]
    assert recipe["critic_formulation"] == family.lower()
    if candidate_id == "stream-bcap":
        assert recipe["optimizer_family"] == "dualnorm"
    assert (tmp_path / "reports/forge" / receipt["artifacts"]["goal_gif"]).is_file()
    assert not (workspace.reports_dir / "receipts/cpu" / candidate_id).exists()


@pytest.mark.parametrize("candidate_id", FAMILY_PRESETS)
def test_one_recipe_per_candidate_across_tasks(candidate_id):
    candidate = TIER1_CANDIDATES[candidate_id]
    recipes = [_build_recipe(task, candidate, task.steps).to_dict() for task in TIER1_TASKS]
    assert all(recipe == recipes[0] for recipe in recipes)


def test_project_cli_rejects_empty_workspace_and_finds_bootstrap(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "validate"]) == 2
    assert "empty gauntlet" in capsys.readouterr().err
    assert main(["--root", str(tmp_path), "bootstrap"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["tasks"] == len(TIER1_TASKS)
    assert result["views"] == [PRIMARY_VIEW.id]
    assert result["candidates"] == len(TIER1_CANDIDATES)
    workspace = StreamWorkspace(tmp_path)
    assert workspace.queue_root == tmp_path / "runs/forge"
    assert (workspace.reports_dir / "technique-inventory.md").is_file()
    assert not (workspace.reports_dir / "leaderboard.md").exists()


def test_pre_upgrade_receipts_do_not_qualify_against_current_engine(tmp_path):
    workspace = bootstrap_workspace(tmp_path)
    legacy = Path(__file__).resolve().parents[1] / "reports/forge/receipts/cuda/fast-weight-bcap-ns"
    target = workspace.reports_dir / "receipts/cuda/fast-weight-bcap-ns"
    target.mkdir(parents=True)
    for receipt in legacy.glob("*.json"):
        (target / receipt.name).write_bytes(receipt.read_bytes())
    board = workspace.qualify("fast-weight-bcap-ns", PRIMARY_VIEW.id, device="cuda")
    assert board["eligible"] is False
    assert all(outcome["status"] == "NOT_RUN"
               for tier in board["tiers"].values() for outcome in tier["tasks"].values())
