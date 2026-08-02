import pytest

from iac_agent.mapping import MappingRegistry, UnmappedKindError
from iac_agent.models import ResourceRecord


@pytest.fixture(scope="module")
def registry():
    return MappingRegistry()


def test_ec2_kinds_are_mapped(registry):
    assert registry.resolve("ec2.vpc").terraform_type == "aws_vpc"
    assert registry.resolve("ec2.eip").terraform_type == "aws_eip"


def test_import_id_renders_from_record(registry):
    record = ResourceRecord(kind="ec2.subnet", id="subnet-42", region="us-east-1")
    km = registry.resolve("ec2.subnet")
    assert km.render_import_id(record) == "subnet-42"


def test_unknown_kind_raises(registry):
    with pytest.raises(UnmappedKindError):
        registry.resolve("rds.instance")


def test_every_mapped_kind_has_valid_shape(registry):
    for kind in registry.known_kinds():
        km = registry.resolve(kind)
        assert km.terraform_type.startswith("aws_")
        assert "{" in km.import_id  # must reference at least one record field
