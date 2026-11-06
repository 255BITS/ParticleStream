"""Historical recipes get explicit strict descendants without optimization changes."""
import json
from pathlib import Path

import pytest
from particlegan_forge import CandidateSpec
from particlestream.online import validate_execution_policy
from particlestream.tier1 import TIER1_TASKS, _build_recipe

ROOT = Path(__file__).resolve().parents[1]
ADDITIONAL = (
    'online-bcap-stateless', 'online-ka2-stateless',
    'online-fast-weight-bcap', 'online-fast-weight-bcap-ns',
    'online-bcap', 'online-bispf', 'online-bispf-optimized',
    'online-bispf-optimized-cool', 'online-ka2', 'online-ka2-clean',
    'online-ka2-light-anchor',
)


@pytest.mark.parametrize('candidate_id', ADDITIONAL)
def test_unrun_baseline_keeps_global_optimization_recipe(candidate_id):
    folder = ROOT / 'configs/forge/ideas'
    child = CandidateSpec.from_dict(json.loads((folder/f'{candidate_id}.json').read_text()))
    parent = CandidateSpec.from_dict(json.loads((folder/f'{child.parent}.json').read_text()))
    assert child.recipe_preset == parent.recipe_preset
    assert child.recipe_overrides == parent.recipe_overrides
    expected = dict(parent.memory_policy)
    expected.update(replay_policy='none', rehearsal_batch=0)
    assert child.memory_policy == expected
    validate_execution_policy(child)
    recipes = [_build_recipe(task, child, task.steps).to_dict() for task in TIER1_TASKS]
    assert all(recipe == recipes[0] for recipe in recipes)
