"""Core data model: the normalized inventory produced by discovery."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ResourceRecord:
    """One discovered AWS resource, in provider-neutral form.

    ``kind`` is the describer-native identifier (e.g. ``ec2.vpc``); the mapping
    layer resolves it to a Terraform resource type and import-ID format.
    """

    kind: str
    id: str
    region: str
    arn: str | None = None
    name_hint: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    attributes: dict = field(default_factory=dict)


@dataclass
class Inventory:
    account_id: str
    regions: list[str]
    scanned_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    records: list[ResourceRecord] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def from_json(cls, path: Path) -> "Inventory":
        data = json.loads(path.read_text())
        records = [ResourceRecord(**r) for r in data.pop("records", [])]
        return cls(records=records, **data)
