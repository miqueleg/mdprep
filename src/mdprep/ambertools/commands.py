"""Shared AmberTools command result models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mdprep.external.runner import CommandResult


class AmberToolsError(RuntimeError):
    """Raised when an AmberTools command is unavailable or fails."""


@dataclass(frozen=True)
class AmberToolRun:
    command_result: CommandResult
    stdout_path: Path
    stderr_path: Path
    output_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command_result.command),
            "cwd": self.command_result.cwd,
            "returncode": self.command_result.returncode,
            "runtime_seconds": self.command_result.runtime_seconds,
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "output_path": str(self.output_path),
        }

