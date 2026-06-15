"""Small subprocess wrapper with reproducibility metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence
import subprocess


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    runtime_seconds: float


def run_command(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = False,
) -> CommandResult:
    workdir = Path(cwd) if cwd is not None else Path.cwd()
    started = perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=workdir,
        env=dict(env) if env is not None else None,
        timeout=timeout,
        text=True,
        capture_output=True,
        check=False,
    )
    runtime = perf_counter() - started
    result = CommandResult(
        command=tuple(str(part) for part in command),
        cwd=str(workdir),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        runtime_seconds=runtime,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(result.command)}"
        )
    return result

