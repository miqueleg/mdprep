"""Optional external executable discovery."""

from __future__ import annotations

from shutil import which


def which_executable(name: str) -> str | None:
    return which(name)


def optional_executable_report(names: list[str]) -> dict[str, str | None]:
    return {name: which_executable(name) for name in names}

