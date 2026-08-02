"""Describer plugin interface.

A describer knows how to enumerate one AWS service's resources via read-only
API calls and emit normalized ResourceRecords. One module per service.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

from ..models import ResourceRecord

if TYPE_CHECKING:  # boto3 imported lazily so unit tests don't require it
    import boto3


def tags_to_dict(tag_list: list[dict] | None) -> dict[str, str]:
    """Convert the AWS [{Key, Value}] tag shape to a plain dict."""
    return {t["Key"]: t["Value"] for t in tag_list or []}


class Describer(ABC):
    service: str

    @abstractmethod
    def describe(
        self, session: "boto3.Session", region: str
    ) -> Iterator[ResourceRecord]:
        """Yield every resource of this service visible in the region."""
