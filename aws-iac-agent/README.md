# aws-iac-agent

An AI agent that scans AWS for infrastructure not managed by Terraform, generates
best-practice Terraform code for it, and validates the code with
`terraform init` / `validate` / `plan` until the plan is clean.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full design.

## Status: Phase 0 (walking skeleton)

The deterministic backbone, no LLM yet:

- `iac-agent scan` — read-only inventory of EC2/VPC resources (VPCs, subnets,
  IGWs, route tables, security groups, instances, NAT gateways, EIPs)
- reconciliation against existing `.tfstate` files
- Terraform `import` block generation with stable, deduplicated addresses
- `terraform init` + `plan -generate-config-out` to produce provably-correct
  baseline HCL

Phase 1 adds the Claude tool-runner agent that refactors the baseline HCL into
modular, best-practice code and drives the plan-clean verification loop.

## Setup

```bash
cd aws-iac-agent
pip install -e ".[dev]"       # needs Python >= 3.11; terraform >= 1.5 on PATH
```

## Usage

```bash
# 1. Scan an account/region with a read-only profile
iac-agent scan --region us-east-1 --profile readonly-prod

# 2. Generate import blocks + baseline HCL, reconciling against existing state
iac-agent generate --state path/to/terraform.tfstate --workspace workspace/

# Or both at once
iac-agent full-run --region us-east-1 --profile readonly-prod

# Then inspect workspace/generated.tf and run `terraform plan` in workspace/ —
# it should show only the pending imports, with no changes.
```

The agent only ever needs a **read-only** AWS role. It never runs
`terraform apply` — the runner cannot even express it.

## Tests

```bash
pytest
```

Unit tests cover the mapping tables, naming, import emission, and state
reconciliation; none of them require AWS credentials or a network connection.
