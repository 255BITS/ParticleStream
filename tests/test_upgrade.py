"""Installed-wheel compatibility and preservation across the 0.9.7 upgrade."""
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import shutil

import particlegan
import particlegan_forge
from particlegan_forge import CandidateSpec, TierSpec, TierView
from particlegan_forge.tiers import engine_source_digest

from particlestream.tier1 import PRIMARY_VIEW, TIER1_TASKS, bootstrap_workspace


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reports/forge/cohorts/particlegan-0.9.6-view5"


def test_both_public_namespaces_come_from_the_pinned_wheel():
    wheel = distribution("particlegan")
    assert wheel.version == "0.9.7"
    for module in (particlegan, particlegan_forge):
        expected = wheel.locate_file(f"{module.__name__}/__init__.py").resolve()
        assert Path(module.__file__).resolve() == expected


def test_pre_upgrade_qualification_evidence_is_preserved():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    assert len(manifest) == 17
    for relative, digest in manifest.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_unchanged_recipe_requires_fresh_engine_bound_receipts(tmp_path):
    workspace = bootstrap_workspace(tmp_path)
    candidate = "online-feature-prior-bce"
    workspace.register_candidate(CandidateSpec.from_dict(json.loads(
        (ROOT / f"configs/forge/ideas/{candidate}.json").read_text())))
    source = ARCHIVE / "receipts/cuda" / candidate
    target = workspace.reports_dir / "receipts/cuda" / candidate
    shutil.copytree(source, target)
    result = workspace.qualify(candidate, PRIMARY_VIEW.id, device="cuda")
    assert result["eligible"] is False
    for tier in result["tiers"].values():
        for outcome in tier["tasks"].values():
            assert outcome["status"] == "NOT_RUN"
            assert "engine_source_sha256" in outcome["reason"]
    for receipt in source.glob("*.json"):
        binding = json.loads(receipt.read_text())["binding"]
        assert binding["engine_source_sha256"] != engine_source_digest()


def test_stretch_reporting_preserves_the_project_workspace_contract(tmp_path):
    workspace = bootstrap_workspace(tmp_path)
    view = TierView.from_tiers(
        id="upgrade_stretch", goal="upgrade_stretch",
        tiers=[TierSpec(tier=1, required=[TIER1_TASKS[0].id],
                        stretch=[TIER1_TASKS[1].id])],
    )
    workspace.register_view(view)
    result = workspace.qualify("online-bcap-adam", view.id)
    assert result["eligible"] is False
    tier = result["tiers"][1]
    assert tier["required_total"] == tier["stretch_total"] == 1
    paths = workspace.compile_reports(view_id=view.id)
    leaderboard = Path(paths["leaderboard"])
    assert leaderboard.name == "technique-inventory.md"
    assert not (workspace.reports_dir / "leaderboard.md").exists()
    content = leaderboard.read_text()
    assert "## Stretch experiments" in content
    assert "Tier 1 stretch" in content and "Total stretch" in content
    assert all(a.importance == "required" for a in workspace.load_views()[PRIMARY_VIEW.id].assignments)
