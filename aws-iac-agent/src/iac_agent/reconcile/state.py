"""Reconciliation: diff the discovered inventory against existing Terraform state.

Matching is on (terraform_type, provider ID). For the Phase 0 services the
provider ID equals the import ID; services where that doesn't hold get a
custom matcher when their describer lands.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..mapping import MappingRegistry, UnmappedKindError
from ..models import ResourceRecord

ManagedKey = tuple[str, str]  # (terraform_type, provider_id)


def load_managed_ids(state_paths: list[Path]) -> set[ManagedKey]:
    """Collect (type, id) pairs for every managed resource in the given state files.

    Accepts raw .tfstate files or the output of `terraform state pull`
    (both are the same v4 JSON document).
    """
    managed: set[ManagedKey] = set()
    for path in state_paths:
        state = json.loads(Path(path).read_text())
        for resource in state.get("resources", []):
            if resource.get("mode") != "managed":
                continue
            rtype = resource["type"]
            for instance in resource.get("instances", []):
                rid = instance.get("attributes", {}).get("id")
                if rid:
                    managed.add((rtype, rid))
    return managed


def diff_unmanaged(
    records: list[ResourceRecord],
    managed: set[ManagedKey],
    mapping: MappingRegistry,
) -> tuple[list[ResourceRecord], list[ResourceRecord]]:
    """Split records into (unmanaged, skipped_unmapped).

    A record is unmanaged when no state file claims its (terraform_type, id).
    Records whose kind has no mapping yet are returned separately so the scan
    report can surface coverage gaps instead of silently dropping them.
    """
    unmanaged: list[ResourceRecord] = []
    unmapped: list[ResourceRecord] = []
    for record in records:
        try:
            km = mapping.resolve(record.kind)
        except UnmappedKindError:
            unmapped.append(record)
            continue
        if (km.terraform_type, km.render_import_id(record)) not in managed:
            unmanaged.append(record)
    return unmanaged, unmapped
