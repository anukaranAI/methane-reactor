from iac_agent.models import ResourceRecord
from iac_agent.naming import resource_name, slugify


def test_slugify_basic():
    assert slugify("My Prod VPC") == "my_prod_vpc"


def test_slugify_leading_digit_and_symbols():
    assert slugify("3-tier app (v2)") == "_3_tier_app_v2"


def test_slugify_empty_falls_back():
    assert slugify("--") == "resource"


def test_resource_name_prefers_name_tag():
    record = ResourceRecord(
        kind="ec2.vpc",
        id="vpc-123",
        region="us-east-1",
        name_hint="hint",
        tags={"Name": "Core VPC"},
    )
    assert resource_name(record) == "core_vpc"


def test_resource_name_falls_back_to_hint_then_id():
    with_hint = ResourceRecord(
        kind="ec2.security_group", id="sg-1", region="us-east-1", name_hint="web-sg"
    )
    assert resource_name(with_hint) == "web_sg"

    bare = ResourceRecord(kind="ec2.vpc", id="vpc-0a1b", region="us-east-1")
    assert resource_name(bare) == "vpc_0a1b"
