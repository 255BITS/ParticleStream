"""Generated links preserve exact recipes, current evidence and historical scope."""
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil

import pytest

from particlestream.forge import StreamWorkspace
from particlestream.reporting import compile_linked_reports
from particlestream.suite import REGISTRY


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path):
    shutil.copytree(ROOT / "configs/forge", tmp_path / "configs/forge")
    return StreamWorkspace(tmp_path, registry=REGISTRY)


def test_every_technique_recipe_and_count_link_reaches_its_generated_section(workspace):
    families = compile_linked_reports(workspace, view_id="single_stream_stability", device="cpu")
    assert len(families) == 6
    inventory = workspace.reports_dir / "technique-inventory.md"
    text = inventory.read_text()
    active = text.split("## All executable recipes\n", 1)[1].split("## Stretch experiments", 1)[0]
    historical = text.split("## Historical declarations\n", 1)[1].split("## Per-Task Breakdown", 1)[0]
    assert active.count("| `INCOMPLETE` |") == 30
    assert historical.count("| READ_ONLY |") == 14
    assert "`stream-bcap-adam`" not in active
    assert "[0/10*]" in active and "0/20" in active
    assert not (workspace.reports_dir / "leaderboard.md").exists()
    documents = [inventory, *map(Path, families)]
    checked = 0
    for document in documents:
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            path, _, fragment = target.partition("#")
            if path.endswith(".md") and fragment:
                destination = (document.parent / path).resolve()
                assert destination.is_file(), (document, target)
                assert f'<a id="{fragment}"></a>' in destination.read_text(), (document, target)
                checked += 1
    assert checked > 1000


def test_representative_is_one_complete_recipe_and_stretch_passes_do_not_qualify(workspace, monkeypatch):
    original = workspace.qualify
    partial = "online-bcap-adam"
    complete = "online-bcap-balanced"

    def qualify(candidate_id, *args, **kwargs):
        result = deepcopy(original(candidate_id, *args, **kwargs))
        if candidate_id not in {partial, complete}:
            return result
        required_index = 0
        for tier in result["tiers"].values():
            for outcome in tier["tasks"].values():
                if outcome["importance"] == "required":
                    required_index += 1
                    if candidate_id == complete:
                        outcome["status"] = "PASS" if required_index == 1 else "FAIL"
                    elif required_index < 10:
                        outcome["status"] = "PASS"
                else:
                    outcome["status"] = "PASS"
        result["status"] = "FAIL"
        return result

    monkeypatch.setattr(workspace, "qualify", qualify)
    compile_linked_reports(workspace, view_id="single_stream_stability", device="cpu")
    text = (workspace.reports_dir / "technique-inventory.md").read_text()
    representatives = text.split("## Representative recipes\n", 1)[1].split("## All executable recipes", 1)[0]
    assert f"`{complete}`" in representatives
    assert f"`{partial}`" not in representatives
    assert "No measured recipe currently qualifies" in text
    details = (workspace.reports_dir / "families/bcap.md").read_text()
    assert "### Configuration delta from parent" in details
    assert "`memory_policy.replay_policy`" in details
    assert "every one of the last five checks must pass" in details
    assert "Historical, read-only" in details


def test_refresh_progress_is_dated_and_does_not_hide_missing_source_bound_results(workspace):
    audit = {"device": "cpu", "started_at": "2026-11-07T10:00:00+00:00", "completed_at": None,
             "planned_runs": 600, "runs": {"one": {}},
             "controls": {"task": {"oracle": {"verdict": "PASS"}, "broken": {"verdict": "FAIL"}}}}
    workspace.reports_dir.mkdir(parents=True, exist_ok=True)
    (workspace.reports_dir / "refresh-audit.json").write_text(json.dumps(audit))
    compile_linked_reports(workspace, view_id="single_stream_stability", device="cpu")
    text = (workspace.reports_dir / "technique-inventory.md").read_text()
    assert "in progress, 1/600 runs recorded" in text
    assert audit["started_at"] in text
    assert "1 oracle PASS, 1 destructive FAIL" in text
    assert "`INCOMPLETE`" in text and "0/20" in text
