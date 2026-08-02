"""EC2/VPC describer: VPCs, subnets, gateways, route tables, SGs, instances, EIPs.

Deliberately skipped in Phase 0 (they need special Terraform handling):
- the default security group per VPC (aws_default_security_group)
- a VPC's main route table (aws_default_route_table)
Skipped resources are reported so the scan output shows what wasn't covered.
"""

from __future__ import annotations

from typing import Iterator

from ..models import ResourceRecord
from .base import Describer, tags_to_dict


class Ec2Describer(Describer):
    service = "ec2"

    def describe(self, session, region: str) -> Iterator[ResourceRecord]:
        client = session.client("ec2", region_name=region)
        yield from self._vpcs(client, region)
        yield from self._subnets(client, region)
        yield from self._internet_gateways(client, region)
        yield from self._route_tables(client, region)
        yield from self._security_groups(client, region)
        yield from self._instances(client, region)
        yield from self._nat_gateways(client, region)
        yield from self._eips(client, region)

    def _vpcs(self, client, region) -> Iterator[ResourceRecord]:
        for page in client.get_paginator("describe_vpcs").paginate():
            for vpc in page["Vpcs"]:
                yield ResourceRecord(
                    kind="ec2.vpc",
                    id=vpc["VpcId"],
                    region=region,
                    tags=tags_to_dict(vpc.get("Tags")),
                    attributes={
                        "cidr_block": vpc.get("CidrBlock"),
                        "is_default": vpc.get("IsDefault", False),
                    },
                )

    def _subnets(self, client, region) -> Iterator[ResourceRecord]:
        for page in client.get_paginator("describe_subnets").paginate():
            for subnet in page["Subnets"]:
                yield ResourceRecord(
                    kind="ec2.subnet",
                    id=subnet["SubnetId"],
                    region=region,
                    arn=subnet.get("SubnetArn"),
                    tags=tags_to_dict(subnet.get("Tags")),
                    attributes={
                        "vpc_id": subnet.get("VpcId"),
                        "cidr_block": subnet.get("CidrBlock"),
                        "availability_zone": subnet.get("AvailabilityZone"),
                    },
                )

    def _internet_gateways(self, client, region) -> Iterator[ResourceRecord]:
        for page in client.get_paginator("describe_internet_gateways").paginate():
            for igw in page["InternetGateways"]:
                yield ResourceRecord(
                    kind="ec2.internet_gateway",
                    id=igw["InternetGatewayId"],
                    region=region,
                    tags=tags_to_dict(igw.get("Tags")),
                )

    def _route_tables(self, client, region) -> Iterator[ResourceRecord]:
        for page in client.get_paginator("describe_route_tables").paginate():
            for rt in page["RouteTables"]:
                is_main = any(
                    assoc.get("Main") for assoc in rt.get("Associations", [])
                )
                if is_main:
                    continue  # main route tables need aws_default_route_table
                yield ResourceRecord(
                    kind="ec2.route_table",
                    id=rt["RouteTableId"],
                    region=region,
                    tags=tags_to_dict(rt.get("Tags")),
                    attributes={"vpc_id": rt.get("VpcId")},
                )

    def _security_groups(self, client, region) -> Iterator[ResourceRecord]:
        for page in client.get_paginator("describe_security_groups").paginate():
            for sg in page["SecurityGroups"]:
                if sg["GroupName"] == "default":
                    continue  # default SGs need aws_default_security_group
                yield ResourceRecord(
                    kind="ec2.security_group",
                    id=sg["GroupId"],
                    region=region,
                    name_hint=sg.get("GroupName"),
                    tags=tags_to_dict(sg.get("Tags")),
                    attributes={"vpc_id": sg.get("VpcId")},
                )

    def _instances(self, client, region) -> Iterator[ResourceRecord]:
        paginator = client.get_paginator("describe_instances")
        for page in paginator.paginate(
            Filters=[
                {
                    "Name": "instance-state-name",
                    "Values": ["pending", "running", "stopping", "stopped"],
                }
            ]
        ):
            for reservation in page["Reservations"]:
                for inst in reservation["Instances"]:
                    yield ResourceRecord(
                        kind="ec2.instance",
                        id=inst["InstanceId"],
                        region=region,
                        tags=tags_to_dict(inst.get("Tags")),
                        attributes={
                            "instance_type": inst.get("InstanceType"),
                            "vpc_id": inst.get("VpcId"),
                            "subnet_id": inst.get("SubnetId"),
                        },
                    )

    def _nat_gateways(self, client, region) -> Iterator[ResourceRecord]:
        paginator = client.get_paginator("describe_nat_gateways")
        for page in paginator.paginate(
            Filters=[{"Name": "state", "Values": ["pending", "available"]}]
        ):
            for nat in page["NatGateways"]:
                yield ResourceRecord(
                    kind="ec2.nat_gateway",
                    id=nat["NatGatewayId"],
                    region=region,
                    tags=tags_to_dict(nat.get("Tags")),
                    attributes={"subnet_id": nat.get("SubnetId")},
                )

    def _eips(self, client, region) -> Iterator[ResourceRecord]:
        # describe_addresses is not paginated
        for addr in client.describe_addresses()["Addresses"]:
            if "AllocationId" not in addr:
                continue  # EC2-Classic addresses are not importable as aws_eip
            yield ResourceRecord(
                kind="ec2.eip",
                id=addr["AllocationId"],
                region=region,
                tags=tags_to_dict(addr.get("Tags")),
                attributes={"public_ip": addr.get("PublicIp")},
            )
