"""iac-agent CLI — Phase 0 walking skeleton.

Commands:
  scan       Inventory AWS resources into scan/inventory.json (read-only).
  generate   Reconcile against existing state, emit import blocks, and run
             terraform init + plan -generate-config-out to produce baseline HCL.
  full-run   scan + generate in one shot.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .mapping import MappingRegistry
from .models import Inventory


def _cmd_scan(args: argparse.Namespace) -> int:
    from .discovery import scan

    inventory = scan(
        regions=args.region, services=args.service or None, profile=args.profile
    )
    out = Path(args.out)
    inventory.to_json(out)
    counts = Counter(r.kind for r in inventory.records)
    print(f"Scanned account {inventory.account_id} in {', '.join(inventory.regions)}")
    for kind, count in sorted(counts.items()):
        print(f"  {kind:30} {count}")
    print(f"{len(inventory.records)} resources -> {out}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from .generate import TerraformRunner, write_workspace
    from .reconcile import diff_unmanaged, load_managed_ids

    inventory = Inventory.from_json(Path(args.inventory))
    mapping = MappingRegistry()
    managed = load_managed_ids([Path(p) for p in args.state])
    unmanaged, unmapped = diff_unmanaged(inventory.records, managed, mapping)

    print(
        f"{len(inventory.records)} discovered, {len(managed)} in existing state, "
        f"{len(unmanaged)} unmanaged, {len(unmapped)} without mapping"
    )
    for record in unmapped:
        print(f"  skipped (no mapping): {record.kind} {record.id}")
    if not unmanaged:
        print("Nothing to generate.")
        return 0

    if len(inventory.regions) != 1:
        print(
            "Phase 0 generates one workspace per region; re-run scan per region.",
            file=sys.stderr,
        )
        return 2

    workspace = Path(args.workspace)
    write_workspace(workspace, unmanaged, mapping, region=inventory.regions[0])
    print(f"Wrote {len(unmanaged)} import blocks -> {workspace}/imports.tf")

    if args.no_terraform:
        return 0

    runner = TerraformRunner(workspace)
    for step in (runner.init, runner.generate_config):
        result = step()
        print(f"$ {' '.join(result.command)}")
        if not result.ok:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            return result.returncode
    print(f"Baseline HCL generated -> {workspace}/generated.tf")
    print("Review it, then run `terraform plan` — it should show imports only.")
    return 0


def _cmd_full_run(args: argparse.Namespace) -> int:
    rc = _cmd_scan(args)
    if rc != 0:
        return rc
    args.inventory = args.out
    return _cmd_generate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iac-agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_scan_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--region", action="append", required=True)
        p.add_argument("--profile", default=None, help="AWS profile (read-only role)")
        p.add_argument("--service", action="append", help="Limit to these services")
        p.add_argument("--out", default="scan/inventory.json")

    def add_generate_args(p: argparse.ArgumentParser, needs_inventory: bool) -> None:
        if needs_inventory:
            p.add_argument("--inventory", default="scan/inventory.json")
        p.add_argument(
            "--state",
            action="append",
            default=[],
            help="Existing .tfstate file(s) to reconcile against (repeatable)",
        )
        p.add_argument("--workspace", default="workspace")
        p.add_argument(
            "--no-terraform",
            action="store_true",
            help="Only write import blocks; skip terraform init/plan",
        )

    p_scan = sub.add_parser("scan", help="Inventory AWS resources")
    add_scan_args(p_scan)
    p_scan.set_defaults(func=_cmd_scan)

    p_gen = sub.add_parser("generate", help="Emit imports + baseline HCL")
    add_generate_args(p_gen, needs_inventory=True)
    p_gen.set_defaults(func=_cmd_generate)

    p_full = sub.add_parser("full-run", help="scan + generate")
    add_scan_args(p_full)
    add_generate_args(p_full, needs_inventory=False)
    p_full.set_defaults(func=_cmd_full_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
