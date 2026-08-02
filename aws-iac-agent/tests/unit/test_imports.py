from iac_agent.generate.imports import emit_import_blocks, write_workspace
from iac_agent.mapping import MappingRegistry
from iac_agent.models import ResourceRecord


def _vpc(vpc_id: str, name: str | None = None) -> ResourceRecord:
    return ResourceRecord(
        kind="ec2.vpc",
        id=vpc_id,
        region="us-east-1",
        tags={"Name": name} if name else {},
    )


def test_emit_single_block():
    out = emit_import_blocks([_vpc("vpc-123", "core")], MappingRegistry())
    assert "import {" in out
    assert "to = aws_vpc.core" in out
    assert 'id = "vpc-123"' in out


def test_colliding_names_are_deduplicated():
    out = emit_import_blocks(
        [_vpc("vpc-1", "app"), _vpc("vpc-2", "app")], MappingRegistry()
    )
    assert "to = aws_vpc.app\n" in out
    assert "to = aws_vpc.app_2\n" in out


def test_same_name_different_types_do_not_collide():
    records = [
        _vpc("vpc-1", "core"),
        ResourceRecord(
            kind="ec2.subnet", id="subnet-1", region="us-east-1", tags={"Name": "core"}
        ),
    ]
    out = emit_import_blocks(records, MappingRegistry())
    assert "to = aws_vpc.core\n" in out
    assert "to = aws_subnet.core\n" in out


def test_write_workspace(tmp_path):
    write_workspace(
        tmp_path / "ws", [_vpc("vpc-1", "core")], MappingRegistry(), region="eu-west-1"
    )
    imports = (tmp_path / "ws" / "imports.tf").read_text()
    versions = (tmp_path / "ws" / "versions.tf").read_text()
    assert "aws_vpc.core" in imports
    assert 'region = "eu-west-1"' in versions
    assert "required_version" in versions
