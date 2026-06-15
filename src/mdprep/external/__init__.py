"""External command helpers."""

from __future__ import annotations

from mdprep.external.discovery import optional_executable_report, which_executable
from mdprep.external.runner import CommandResult, run_command

__all__ = [
    "CommandResult",
    "optional_executable_report",
    "run_command",
    "which_executable",
]

