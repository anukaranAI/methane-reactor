"""Mapping layer: describer-native resource kinds -> Terraform types + import IDs.

The tables live as YAML data (one file per AWS service) so coverage can grow
without code changes and each entry is unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..models import ResourceRecord

TABLES_DIR = Path(__file__).parent / "tables"


@dataclass(frozen=True)
class KindMapping:
    kind: str
    terraform_type: str
    import_id: str  # format template rendered with the record's fields

    def render_import_id(self, record: ResourceRecord) -> str:
        return self.import_id.format(
            id=record.id, region=record.region, **record.attributes
        )


class MappingRegistry:
    def __init__(self, tables_dir: Path = TABLES_DIR):
        self._by_kind: dict[str, KindMapping] = {}
        for table in sorted(tables_dir.glob("*.yaml")):
            data = yaml.safe_load(table.read_text())
            for kind, spec in data.get("kinds", {}).items():
                self._by_kind[kind] = KindMapping(
                    kind=kind,
                    terraform_type=spec["terraform_type"],
                    import_id=spec.get("import_id", "{id}"),
                )

    def resolve(self, kind: str) -> KindMapping:
        try:
            return self._by_kind[kind]
        except KeyError:
            raise UnmappedKindError(kind) from None

    def known_kinds(self) -> list[str]:
        return sorted(self._by_kind)


class UnmappedKindError(KeyError):
    def __init__(self, kind: str):
        super().__init__(f"No Terraform mapping for resource kind {kind!r}")
        self.kind = kind
