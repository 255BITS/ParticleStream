"""Run public Forge CLI comparisons sequentially and independently audit evidence.

Use --commit --push to publish each completed candidate. Creating --stop-file
pauses at the next candidate boundary after evidence has been audited/saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def audit_candidate(workspace, candidate_id, device):
    from PIL import Image
    from particlegan_forge import grade_sustained_observations
    tasks = workspace.load_tasks()
    outcomes, recipes = {}, []
    qualification = workspace.qualify(candidate_id, 'single_stream_stability', device=device)
    qualified = {task_id: outcome for tier in qualification['tiers'].values()
                 for task_id, outcome in tier['tasks'].items()}
    for task in tasks.values():
        path = workspace.reports_dir / 'receipts' / device / candidate_id / f'{task.id}.json'
        receipt = json.loads(path.read_text())
        assert receipt['error'] is None, receipt['error']
        assert receipt['seed'] == 0 and receipt['completed_steps'] == task.steps == 480
        assert receipt['unintended_rng_deviations'] == 0
        assert [row['step'] for row in receipt['observations']] == task.observation_steps()
        grade = grade_sustained_observations(
            task, receipt['observations'], all_finite=receipt['all_finite'],
            unintended_rng_deviations=0, completed_steps=480,
        )
        assert grade['status'] == receipt['verdict'] == qualified[task.id]['status']
        assert grade['terminal_passing'] == receipt['terminal_passing']
        execution = receipt['artifacts']['online_execution']
        warmup = task.host_definition['online_contract']['warmup_arrivals']
        assert execution['arrivals'] == 480 + warmup
        assert execution['updates'] == execution['live_predictions'] == 480
        assert execution['max_real_batch'] == execution['max_fake_batch'] == 1
        assert execution['replayed_examples'] == 0
        assert execution['gradient_accumulation_steps'] == 1
        gif = workspace.reports_dir / receipt['artifacts']['goal_gif']
        assert hashlib.sha256(gif.read_bytes()).hexdigest() == receipt['artifacts']['gif_sha256']
        with Image.open(gif) as image:
            assert image.n_frames == 24 and image.size == (900, 300)
            for index in range(image.n_frames):
                image.seek(index)
                image.load()
        recipes.append(receipt['artifacts']['resolved_recipe'])
        outcomes[task.id] = {
            'status': grade['status'], 'terminal_passing': receipt['terminal_passing'],
            'all_finite': receipt['all_finite'],
            'final_metrics': receipt['observations'][-1]['metrics'],
        }
    assert all(recipe == recipes[0] for recipe in recipes), 'per-task recipe selection'
    return {'passed_tasks': sum(result['status'] == 'PASS' for result in outcomes.values()),
            'tasks': outcomes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--candidate', action='append', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--summary', type=Path, default=Path('reports/forge/unrun-baselines-summary.json'))
    parser.add_argument('--stop-file', type=Path, default=Path('runs/forge/stop-comparison'))
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--commit', action='store_true')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from particlestream.forge import StreamWorkspace
    from particlestream.tier1 import REGISTRY
    workspace = StreamWorkspace(root, registry=REGISTRY)
    workspace.validate()
    summary_path = root / args.summary
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
        'protocol_seed': 0, 'task_count': len(workspace.load_tasks()), 'candidates': {},
    }
    summary['deferred_candidates'] = sorted(set(summary.get('deferred_candidates', [])) | set(args.candidate))
    for candidate_id in args.candidate:
        if (root / args.stop_file).exists():
            print(json.dumps({'paused_before': candidate_id, 'stop_file': str(args.stop_file)}), flush=True)
            break
        start = time.monotonic()
        if not args.audit_only:
            log = root / 'runs/forge' / f'{candidate_id}-comparison.json'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('w') as output:
                subprocess.run([sys.executable, '-m', 'particlestream.forge', 'run', candidate_id,
                                '--device', args.device], cwd=root, stdout=output,
                               stderr=subprocess.STDOUT, check=True)
        result = audit_candidate(workspace, candidate_id, args.device)
        result['elapsed_seconds'] = round(time.monotonic() - start, 3)
        result['source_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        summary['candidates'][candidate_id] = result
        summary['deferred_candidates'] = [candidate for candidate in summary['deferred_candidates']
                                          if candidate != candidate_id]
        if not summary['deferred_candidates']:
            summary.pop('deferred_reason', None)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
        if args.commit or args.push:
            paths = [str(summary_path), f'reports/forge/receipts/{args.device}/{candidate_id}',
                     *[str(path) for path in (workspace.reports_dir/'gifs').glob(f'{candidate_id}_*.gif')],
                     'reports/forge/EXPERIMENT_MEMORY.md', 'reports/forge/technique-inventory.md']
            subprocess.run(['git', 'add', *paths], cwd=root, check=True)
            subprocess.run(['git', 'commit', '-m', f'Record strict Tier1 evidence for {candidate_id}'], cwd=root, check=True)
            if args.push:
                subprocess.run(['git', 'push'], cwd=root, check=True)
        print(json.dumps({'candidate': candidate_id, 'passed_tasks': result['passed_tasks'],
                          'task_count': len(result['tasks']), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
