from pathlib import Path


BANNED_STRINGS = ["openbabel", "OpenBabel", "pybel", "obabel"]
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
}


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_DIRS:
        return True
    if any(part.startswith(".") for part in path.parts if part not in {".", ".."}):
        return True
    if any(part.endswith(".egg-info") for part in path.parts):
        return True
    return False


def test_banned_toolkit_strings_do_not_appear_outside_this_test():
    root = Path(__file__).resolve().parents[1]
    this_file = Path(__file__).resolve()
    offenders: list[tuple[Path, str]] = []

    for path in root.rglob("*"):
        if path.resolve() == this_file:
            continue
        if path.is_dir() or should_skip(path.relative_to(root)):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for banned in BANNED_STRINGS:
            if banned in text:
                offenders.append((path.relative_to(root), banned))

    assert offenders == []

