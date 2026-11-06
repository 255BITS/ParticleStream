"""ParticleStream's paths and CLI on the public particlegan_forge Tier API.

The upstream workspace defaults to its own namespaced directories. This project
retains its established configs/forge, reports/forge and runs/forge contracts.
Training, grading, RNG auditing and receipt binding remain upstream-owned.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from particlegan_forge import ForgeWorkspace, atomic_text, json_safe


class StreamWorkspace(ForgeWorkspace):
    def __init__(self, root, *, registry=None, queue_root=None):
        root = Path(root).resolve()
        super().__init__(root, registry=registry, queue_root=queue_root or root / "runs/forge")

    @property
    def configs_dir(self):
        return self.root / "configs/forge"

    @property
    def reports_dir(self):
        return self.root / "reports/forge"

    def validate(self):
        result = super().validate()
        if not result["tasks"] or not result["views"] or not result["candidates"]:
            raise ValueError("empty gauntlet; run particlestream bootstrap first")
        from .tier1 import IMPLEMENTATION, _build_recipe
        for task in self.load_tasks().values():
            self.registry.get_adapter(task.adapter)
            if task.host_definition.get("implementation") != IMPLEMENTATION:
                raise ValueError("stale task implementation; run particlestream bootstrap")
            for candidate in self.load_candidates().values():
                _build_recipe(task, candidate, task.steps)
        return result

    def compile_reports(self, **kwargs):
        paths = super().compile_reports(**kwargs)
        generated = Path(paths["leaderboard"])
        canonical = self.reports_dir / "technique-inventory.md"
        content = generated.read_text().replace(
            "reports/particlegan_forge/receipts/", "reports/forge/receipts/"
        )
        atomic_text(canonical, content.rstrip() + "\n")
        generated.unlink()
        paths["leaderboard"] = str(canonical)
        memory = Path(paths["experiment_memory"])
        content = memory.read_text().replace(
            "](leaderboard.md)", "](technique-inventory.md)"
        )
        if (self.reports_dir / "research-notes.md").is_file():
            content += "\nInterpretation and historical evidence: [research notes](research-notes.md).\n"
        atomic_text(memory, content)
        return paths


def main(argv=None):
    from .tier1 import PRIMARY_VIEW, REGISTRY, bootstrap_workspace

    parser = argparse.ArgumentParser(description="ParticleStream gauntlet (particlegan_forge)")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--provider", choices=("particlestream.tier1",),
                        help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("bootstrap")
    commands.add_parser("validate")
    retier = commands.add_parser("retier")
    retier.add_argument("view")
    retier.add_argument("--task", required=True)
    retier.add_argument("--tier", required=True, type=int)
    retier.add_argument("--reason", required=True)
    for name in ("run", "compare", "board", "compile"):
        command = commands.add_parser(name)
        command.add_argument("--view", default=PRIMARY_VIEW.id)
        command.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
        if name == "run":
            command.add_argument("candidate")
            command.add_argument("--through-tier", type=int, default=1)
        elif name == "compare":
            command.add_argument("--candidate", action="append",
                                 help="Repeat to select candidates; default runs all families, BISPF first")
    args = parser.parse_args(argv)
    workspace = StreamWorkspace(args.root, registry=REGISTRY)
    try:
        if args.command == "bootstrap":
            workspace = bootstrap_workspace(args.root)
            result = workspace.validate()
        elif args.command == "validate":
            result = workspace.validate()
        elif args.command == "board":
            result = {cid: workspace.qualify(cid, args.view, device=args.device)
                      for cid in sorted(workspace.load_candidates())}
        elif args.command == "compile":
            result = workspace.compile_reports(view_id=args.view, device=args.device)
        elif args.command == "retier":
            result = workspace.retier(args.view, task_id=args.task, new_tier=args.tier,
                                       reason=args.reason).to_dict()
        elif args.command == "compare":
            workspace.validate()
            selected = args.candidate or sorted(
                (cid for cid in workspace.load_candidates() if cid.startswith("stream-")),
                key=lambda cid: ("bispf" not in cid, cid),
            )
            result = {}
            for candidate in selected:
                result[candidate] = workspace.run_tier(candidate, args.view, device=args.device)
                workspace.compile_reports(view_id=args.view, device=args.device)
        else:
            workspace.validate()
            result = workspace.run_tier(args.candidate, args.view,
                                        through_tier=args.through_tier, device=args.device)
            workspace.compile_reports(view_id=args.view, device=args.device)
        print(json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as error:
        import sys
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
