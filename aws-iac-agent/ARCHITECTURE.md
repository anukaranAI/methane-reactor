# AWS → Terraform IaC Agent — Architecture

An AI agent that **scans an AWS account, finds infrastructure not managed by Terraform, generates production-quality Terraform code for it, and validates that code end-to-end** (`terraform init` → `validate` → `plan`) until the plan is clean — meaning the generated code exactly matches what exists in AWS.

---

## 1. Goals & Non-Goals

### Goals
1. **Discover** all (supported) resources in one or more AWS accounts/regions.
2. **Reconcile** discovered resources against existing Terraform state to isolate *unmanaged* infrastructure.
3. **Generate** Terraform code for unmanaged resources using best practices: modules, variables, consistent naming/tagging, remote state, provider version pinning.
4. **Validate autonomously**: run `terraform init`, `terraform validate`, `terraform plan` in a loop, fixing errors until the plan shows **zero changes** (proof the code matches reality).
5. **Deliver safely**: output lands as a Git branch / pull request for human review — the agent never mutates infrastructure.

### Non-Goals (v1)
- Running `terraform apply` (explicitly out of scope — the agent is read-only against AWS).
- Multi-cloud (Azure/GCP) — the design leaves room for it, but v1 is AWS only.
- Cost optimization / rightsizing recommendations (possible later phase).

---

## 2. The Key Insight: Don't Make the LLM Guess HCL

The most common failure mode of "AI generates Terraform" projects is asking the LLM to write HCL from an API dump — it hallucinates arguments, misses computed-only attributes, and produces plans full of diffs.

Terraform ≥ 1.5 solves the hard part natively:

```
import block  +  terraform plan -generate-config-out=generated.tf
```

Terraform itself emits **provably correct** HCL for any resource you can address. So the pipeline is:

1. **Deterministic code** (not the LLM) scans AWS and emits `import` blocks.
2. **Terraform** generates the raw, correct-but-ugly HCL.
3. **The LLM agent** does what it's actually good at: refactoring that raw HCL into clean, modular, DRY, best-practice code — extracting variables, grouping into modules, removing provider-default noise, adding tags — and then **re-running `terraform plan` after every edit** to prove it didn't change semantics.

The `plan` loop is the agent's ground truth. The definition of done is: *all imports resolve, and `terraform plan` reports no changes.*

---

## 3. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                            ORCHESTRATOR (CLI)                          │
│                     `iac-agent scan --profile prod`                    │
└──────┬─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐   ┌──────────────────┐   ┌───────────────────────────┐
│ 1. DISCOVERY │──▶│ 2. RECONCILIATION │──▶│ 3. IMPORT-BLOCK GENERATOR │
│  (boto3,     │   │  (diff vs. exist- │   │  (deterministic: emits    │
│  read-only)  │   │   ing tfstate)    │   │   import {} blocks)       │
└──────────────┘   └──────────────────┘   └────────────┬──────────────┘
                                                       │
                                                       ▼
                                        ┌──────────────────────────────┐
                                        │ 4. BASELINE HCL GENERATION   │
                                        │  terraform plan              │
                                        │    -generate-config-out      │
                                        └────────────┬─────────────────┘
                                                     │ raw HCL
                                                     ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 5. LLM REFACTORING AGENT  (Claude, agentic tool loop)                 │
│                                                                       │
│   Tools: read_file · write_file · run_terraform (init/validate/plan)  │
│          run_linter (tflint) · run_security_scan (checkov/trivy)      │
│                                                                       │
│   Loop:  refactor → validate → plan → fix → …until plan is clean      │
│          and linters/security scanners pass                           │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
                    ┌───────────────────────────┐
                    │ 6. DELIVERY               │
                    │  git branch + PR with     │
                    │  summary, plan output,    │
                    │  scan report              │
                    └───────────────────────────┘
```

---

## 4. Component Design

### 4.1 Discovery Engine (deterministic, Python + boto3)

Responsible for producing a complete inventory of resources. Three complementary strategies, in order of preference:

| Strategy | What it gives you | Notes |
|---|---|---|
| **Resource Groups Tagging API** (`resourcegroupstaggingapi:GetResources`) | Fast, cross-service inventory of *taggable* resources with ARNs | Misses untaggable resources (e.g., route table associations) |
| **AWS Config / Resource Explorer** (if enabled) | Rich inventory incl. relationships and history | Best source when the account has it enabled; don't require it |
| **Per-service boto3 describers** | Everything else (IAM inline policies, SG rules, listeners…) | A plugin registry: one small describer module per service, prioritized by coverage (EC2/VPC, IAM, S3, RDS, Lambda, etc.) |

Output: a normalized `Inventory` — `{resource_type, terraform_type, id, arn, region, tags, raw_attributes}` — written to `scan/inventory.json`.

**Credentials**: a dedicated read-only IAM role (start from AWS-managed `ReadOnlyAccess` or the tighter `ViewOnlyAccess`, ideally scoped down to the services in the describer registry). The agent process never receives write-capable credentials — safety is enforced by IAM, not by prompting.

### 4.2 Reconciliation

- Pull existing state: parse `terraform state pull` from each known workspace/backend (config file lists the state locations), or accept `--no-existing-state` for greenfield accounts.
- Match by provider ID/ARN. Everything in the inventory but not in any state file is **unmanaged** → candidate for generation.
- Also detect the inverse (in state but gone from AWS) and report it as drift — useful signal, but not acted on in v1.

### 4.3 Import-Block Generator (deterministic)

For each unmanaged resource, map `(service, resource kind, id)` → `(terraform resource type, import ID format)` and emit:

```hcl
import {
  to = aws_s3_bucket.assets_prod
  id = "assets-prod-us-east-1"
}
```

This mapping table (AWS API shape → Terraform type + import-ID syntax) is the main piece of domain knowledge the project owns; it lives as data (YAML), one file per service, and is unit-testable. Resource *naming* (the `to =` address) is generated from tags/Name/ID with a deterministic slugifier; the LLM may rename later during refactoring (renames are safe pre-import since nothing is in state yet).

### 4.4 Baseline HCL Generation

Run in a sandboxed workspace:

```bash
terraform init
terraform plan -generate-config-out=generated.tf
```

Handle the known rough edges in code, not in the LLM: resources whose generated config includes write-only/computed attributes that fail `plan`, provider version pinning, region/provider aliasing for multi-region scans.

### 4.5 LLM Refactoring Agent

This is the "AI" in the system, built as a **Claude API tool-use agent** (see §5). Its job, with the raw `generated.tf` + inventory as input:

1. **Structure** — split monolithic generated code into a conventional layout (`network.tf`, `compute.tf`, `iam.tf`, …) or into reusable modules when it sees repetition (e.g., 12 nearly-identical SGs → one module + `for_each`).
2. **Parameterize** — extract environment-specific literals into `variables.tf` with types, descriptions, and sane defaults; produce `terraform.tfvars`.
3. **De-noise** — remove attributes that merely restate provider defaults (a curated allowlist guards against removing load-bearing ones — every removal must still produce a clean plan).
4. **Best practices** — `versions.tf` with pinned provider/terraform versions, remote-backend stanza (S3 backend, encrypted, with locking) emitted commented-out for the human to activate, `default_tags`, consistent naming.
5. **Verify after every change** — call `run_terraform("validate")` and `run_terraform("plan")`; a refactor is only kept if the plan stays clean. `tflint` and `checkov`/`trivy` findings are fed back into the loop; security findings that can't be auto-fixed (they'd change real infrastructure) are written to `REVIEW.md` instead.

The agent's tools are **narrow and typed**, not a raw shell:

| Tool | Why it's a dedicated tool |
|---|---|
| `read_file` / `write_file` / `list_files` | Confined to the workspace dir (path traversal guarded) |
| `run_terraform(command: init\|validate\|plan)` | Enum-constrained — `apply`/`destroy` are unrepresentable |
| `run_linter()` / `run_security_scan()` | Fixed commands, parsed structured output |
| `get_resource_detail(arn)` | Read-only lookup into the inventory for context |

### 4.6 Delivery

- Everything is written into a Git repo (new or existing IaC repo): `envs/<account>/<region>/` layout.
- The agent opens a branch + PR containing: the generated code, the final clean `terraform plan` output as proof, the scan report (counts per service, skipped/unsupported resources), and `REVIEW.md` (security findings, judgment calls, suggested follow-ups).
- A human reviews and merges; running the actual `terraform apply` (which executes the imports and takes ownership of the resources in state) is a deliberate human step.

---

## 5. Agent Implementation Choice

Two viable ways to build §4.5, both hosted on our own infra:

| Option | What it is | Fit |
|---|---|---|
| **Anthropic SDK Tool Runner** (`client.beta.messages.tool_runner`, Python `@beta_tool`) | The SDK drives the request→tool→result loop over *our* typed tools | **Recommended.** We want exactly the narrow tool surface above — no general bash, no arbitrary file access outside the workspace. Per-turn hooks give us logging, retry, and result inspection for free. |
| **Claude Agent SDK** (Claude Code as a library) | Batteries-included harness with built-in bash/file tools | Faster to prototype, but its built-in Bash/Write tools are broader than we want pointed at cloud credentials and a state file. Good for a spike, not the product. |

**Decision: Claude API + Tool Runner (Python).**

Key API parameters:

- **Model**: `claude-opus-5` — the refactoring/fix loop is agentic coding work, squarely its strength.
- **Thinking**: on by default on Opus 5 (omit the `thinking` param).
- **Effort**: `output_config: {effort: "xhigh"}` for the refactoring loop (the recommended setting for coding/agentic work); sweep down to `high`/`medium` later if evals show quality holds.
- **Streaming** with a generous `max_tokens` (≥ 64K) — refactoring turns are long.
- **Prompt caching**: system prompt (conventions, tool docs, refactoring rules) is frozen and cached; the volatile per-run content (inventory summary, plan output) comes after the cache breakpoint.
- **Refusal handling**: check `stop_reason == "refusal"` before reading content, and enable server-side fallbacks (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`) so a rare classifier false-positive on security-ish content (IAM policies, SG rules) degrades gracefully instead of failing the run.
- **Loop budget**: cap iterations (e.g., 25 tool-loop turns per module) and use `task_budget` (beta) so the model paces itself instead of being cut off mid-refactor.

Skeleton:

```python
import anthropic
from anthropic import beta_tool

client = anthropic.Anthropic()

@beta_tool
def run_terraform(command: str) -> str:
    """Run a terraform command in the workspace.

    Args:
        command: One of "init", "validate", "plan". Other values are rejected.
    """
    assert command in {"init", "validate", "plan"}
    return _exec_terraform(command)   # subprocess, cwd=workspace, timeout, output truncated smartly

runner = client.beta.messages.tool_runner(
    model="claude-opus-5",
    max_tokens=64000,
    output_config={"effort": "xhigh"},
    tools=[read_file, write_file, list_files, run_terraform, run_linter, run_security_scan],
    messages=[{"role": "user", "content": refactor_task_prompt}],
)
for message in runner:
    log_turn(message)
```

---

## 6. Safety Model

| Risk | Mitigation |
|---|---|
| Agent mutates infrastructure | Read-only IAM role; `run_terraform` tool cannot express `apply`/`destroy`; no raw shell tool |
| Secrets leak into generated code | Redaction pass on inventory before it reaches the LLM (mark `SecretString`, key material, connection strings); generated code references data sources / variables marked `sensitive = true`, never literals |
| Bad refactor changes semantics | Every edit must re-pass `terraform plan` with zero diff — enforced by the orchestrator, not trusted to the model |
| State file corruption | Agent works in an isolated workspace with a *local* throwaway state; the real backend is only touched by the human after PR merge |
| Runaway cost / infinite loops | Turn caps + `task_budget`; per-run token budget alerting |
| Over-broad scan blast radius | Scans are per-account/per-region, explicit allowlist of services in v1 |

---

## 7. Repository Layout (target)

```
aws-iac-agent/
├── ARCHITECTURE.md            ← this document
├── pyproject.toml
├── src/iac_agent/
│   ├── cli.py                 # entrypoint: scan / generate / full-run
│   ├── discovery/             # boto3 describers, one module per service
│   ├── reconcile/             # tfstate parsing + diffing
│   ├── mapping/               # YAML: AWS kind → TF type + import-ID format
│   ├── generate/              # import-block emitter, terraform runner
│   ├── agent/                 # Claude tool-runner loop, tools, prompts
│   │   ├── tools.py
│   │   ├── prompts/system.md
│   │   └── loop.py
│   └── delivery/              # git branch / PR creation, reports
├── tests/
│   ├── unit/                  # mapping tables, slugifier, reconciler
│   └── e2e/                   # LocalStack-backed: seed resources → run pipeline → assert clean plan
└── examples/
```

**Testing strategy**: unit tests for all deterministic parts; end-to-end tests against **LocalStack** (seed known resources, run the whole pipeline, assert the generated code plans clean); a small eval set of real-world-ish accounts to measure the LLM refactoring quality (plan-clean rate, lint pass rate, human review score).

---

## 8. Phased Roadmap

| Phase | Deliverable | Scope |
|---|---|---|
| **0 — Walking skeleton** (1–2 wks) | CLI that scans one service (VPC/EC2), emits import blocks, runs `plan -generate-config-out`, commits raw HCL | No LLM yet — proves the deterministic backbone |
| **1 — Agent loop** | Claude tool-runner refactors the generated HCL; plan-clean gate; PR delivery | Single region, ~5 core services (VPC, EC2, S3, IAM, RDS) |
| **2 — Coverage & quality** | 20+ services via describer plugins; module extraction; tflint + checkov in the loop; LocalStack e2e suite | |
| **3 — Operations** | Scheduled drift-detection mode (re-scan, report new unmanaged resources), multi-account via role assumption, Slack/PR notifications | |
| **4 — Nice-to-haves** | Cost annotations, module registry suggestions, Terragrunt/Terraform-stacks output option | |

---

## 9. Open Questions (decide before Phase 1)

1. **Target IaC repo**: does generated code land in a new repo per account, or an existing central IaC monorepo? (Affects delivery + backend config.)
2. **Terraform vs OpenTofu**: pipeline is identical; pick one for CI images.
3. **Which services first**: the priority list for describer plugins should come from what the real account actually contains — run Phase 0's tag-API scan to get the histogram.
4. **State backends in use today**: reconciliation needs the list of existing state locations (S3 buckets/keys, TFC workspaces).
