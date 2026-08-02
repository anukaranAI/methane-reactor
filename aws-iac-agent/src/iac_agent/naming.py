"""Deterministic Terraform resource-address naming for discovered resources."""

from __future__ import annotations

import re

from .models import ResourceRecord

_INVALID = re.compile(r"[^a-z0-9_]+")


def slugify(text: str) -> str:
    """Turn arbitrary text into a valid Terraform identifier."""
    slug = _INVALID.sub("_", text.strip().lower()).strip("_")
    if not slug:
        slug = "resource"
    if slug[0].isdigit():
        slug = f"_{slug}"
    return slug


def resource_name(record: ResourceRecord) -> str:
    """Pick a stable, human-meaningful address name for a resource.

    Preference order: Name tag, describer-provided name hint, raw ID.
    """
    candidate = record.tags.get("Name") or record.name_hint or record.id
    return slugify(candidate)
