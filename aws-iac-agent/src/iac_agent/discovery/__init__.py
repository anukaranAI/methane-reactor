"""Discovery engine: runs registered describers against an AWS session."""

from __future__ import annotations

from ..models import Inventory
from .base import Describer
from .ec2 import Ec2Describer

REGISTRY: dict[str, Describer] = {
    Ec2Describer.service: Ec2Describer(),
}


def scan(
    regions: list[str],
    services: list[str] | None = None,
    profile: str | None = None,
) -> Inventory:
    """Scan the account reachable with the given profile and return an Inventory."""
    import boto3  # lazy: unit tests of other modules don't need it

    session = boto3.Session(profile_name=profile)
    account_id = session.client("sts").get_caller_identity()["Account"]

    selected = services or list(REGISTRY)
    unknown = set(selected) - set(REGISTRY)
    if unknown:
        raise ValueError(
            f"Unknown services: {sorted(unknown)}. Available: {sorted(REGISTRY)}"
        )

    inventory = Inventory(account_id=account_id, regions=regions)
    for region in regions:
        for name in selected:
            inventory.records.extend(REGISTRY[name].describe(session, region))
    return inventory
