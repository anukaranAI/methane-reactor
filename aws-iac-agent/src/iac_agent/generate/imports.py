"""Emit Terraform import blocks and set up the generation workspace."""

from __future__ import annotations

from pathlib import Path

from ..mapping import MappingRegistry
from ..models import ResourceRecord
from ..naming import resource_name

VERSIONS_TF = """\
terraform {{
  required_version = ">= 1.5"

  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }}
  }}
}}

provider "aws" {{
  region = "{region}"
}}
"""


def emit_import_blocks(
    records: list[ResourceRecord], mapping: MappingRegistry
) -> str:
    """Render one import block per record, with unique resource addresses."""
    blocks: list[str] = []
    used: set[str] = set()
    for record in sorted(records, key=lambda r: (r.kind, r.id)):
        km = mapping.resolve(record.kind)
        base = resource_name(record)
        name, n = base, 2
        while f"{km.terraform_type}.{name}" in used:
            name = f"{base}_{n}"
            n += 1
        used.add(f"{km.terraform_type}.{name}")
        blocks.append(
            "import {\n"
            f"  to = {km.terraform_type}.{name}\n"
            f'  id = "{km.render_import_id(record)}"\n'
            "}\n"
        )
    return "\n".join(blocks)


def write_workspace(
    workspace: Path,
    records: list[ResourceRecord],
    mapping: MappingRegistry,
    region: str,
) -> None:
    """Write imports.tf + versions.tf into a fresh generation workspace."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "imports.tf").write_text(emit_import_blocks(records, mapping))
    (workspace / "versions.tf").write_text(VERSIONS_TF.format(region=region))
