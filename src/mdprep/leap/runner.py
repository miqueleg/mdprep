"""tleap subprocess wrapper."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from mdprep.external.discovery import which_executable
from mdprep.external.runner import CommandResult, run_command
from mdprep.leap.log_parser import LeapLogSummary, parse_tleap_log


class TLeapRunError(RuntimeError):
    """Raised when tleap cannot be launched."""


@dataclass(frozen=True)
class TLeapRun:
    command_result: CommandResult
    input_path: Path
    log_path: Path
    summary: LeapLogSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command_result.command),
            "cwd": self.command_result.cwd,
            "returncode": self.command_result.returncode,
            "runtime_seconds": self.command_result.runtime_seconds,
            "input_path": str(self.input_path),
            "log_path": str(self.log_path),
            "summary": self.summary.to_dict(),
        }


def run_tleap(
    input_path: str | Path,
    *,
    work_dir: str | Path,
    executable: str = "tleap",
) -> TLeapRun:
    exe = which_executable(executable)
    if exe is None:
        raise TLeapRunError(f"AmberTools executable not found: {executable}")
    work = Path(work_dir)
    script = Path(input_path)
    result = run_command([exe, "-f", script.name], cwd=work)
    generated_log = work / "leap.log"
    log_path = work / "tleap.log"
    if generated_log.exists():
        shutil.copyfile(generated_log, log_path)
    else:
        log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    return TLeapRun(
        command_result=result,
        input_path=script,
        log_path=log_path,
        summary=parse_tleap_log(log_path, returncode=result.returncode),
    )
