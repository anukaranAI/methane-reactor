"""Sandboxed Terraform runner.

Only read-only commands are representable: init, validate, plan, and
plan -generate-config-out. There is deliberately no way to express apply or
destroy — that invariant also holds for the future LLM agent tool, which wraps
this runner.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ALLOWED_COMMANDS = ("init", "validate", "plan")


@dataclass
class TerraformResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class TerraformNotFoundError(RuntimeError):
    pass


class TerraformRunner:
    def __init__(self, workspace: Path, timeout: int = 600):
        self.workspace = Path(workspace)
        self.timeout = timeout
        if shutil.which("terraform") is None:
            raise TerraformNotFoundError(
                "terraform binary not found on PATH (need >= 1.5)"
            )

    def run(self, command: str, *extra_args: str) -> TerraformResult:
        if command not in ALLOWED_COMMANDS:
            raise ValueError(
                f"Command {command!r} not allowed; only {ALLOWED_COMMANDS}"
            )
        argv = ["terraform", command, "-no-color", *extra_args]
        proc = subprocess.run(
            argv,
            cwd=self.workspace,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return TerraformResult(argv, proc.returncode, proc.stdout, proc.stderr)

    def init(self) -> TerraformResult:
        return self.run("init", "-input=false")

    def validate(self) -> TerraformResult:
        return self.run("validate")

    def plan(self) -> TerraformResult:
        return self.run("plan", "-input=false")

    def generate_config(self, out_file: str = "generated.tf") -> TerraformResult:
        """Run the config-generation plan that turns import blocks into HCL."""
        return self.run("plan", "-input=false", f"-generate-config-out={out_file}")
