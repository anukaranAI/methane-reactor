import json

from iac_agent.mapping import MappingRegistry
from iac_agent.models import ResourceRecord
from iac_agent.reconcile import diff_unmanaged, load_managed_ids


def _state_file(tmp_path, resources):
    state = {"version": 4, "resources": resources}
    path = tmp_path / "terraform.tfstate"
    path.write_text(json.dumps(state))
    return path


def test_load_managed_ids(tmp_path):
    path = _state_file(
        tmp_path,
        [
            {
                "mode": "managed",
                "type": "aws_vpc",
                "name": "core",
                "instances": [{"attributes": {"id": "vpc-managed"}}],
            },
            {
                "mode": "data",  # data sources are not managed resources
                "type": "aws_vpc",
                "name": "lookup",
                "instances": [{"attributes": {"id": "vpc-data"}}],
            },
        ],
    )
    assert load_managed_ids([path]) == {("aws_vpc", "vpc-managed")}


def test_diff_splits_unmanaged_and_unmapped():
    records = [
        ResourceRecord(kind="ec2.vpc", id="vpc-managed", region="us-east-1"),
        ResourceRecord(kind="ec2.vpc", id="vpc-new", region="us-east-1"),
        ResourceRecord(kind="rds.instance", id="db-1", region="us-east-1"),
    ]
    unmanaged, unmapped = diff_unmanaged(
        records, {("aws_vpc", "vpc-managed")}, MappingRegistry()
    )
    assert [r.id for r in unmanaged] == ["vpc-new"]
    assert [r.id for r in unmapped] == ["db-1"]


def test_empty_state_means_everything_unmanaged():
    records = [ResourceRecord(kind="ec2.vpc", id="vpc-1", region="us-east-1")]
    unmanaged, unmapped = diff_unmanaged(records, set(), MappingRegistry())
    assert len(unmanaged) == 1 and not unmapped
