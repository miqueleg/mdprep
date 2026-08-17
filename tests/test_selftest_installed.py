from io import StringIO

from rich.console import Console

from mdprep.workflows import selftest as selftest_module


def test_selftest_uses_bundled_resources_outside_source_checkout(monkeypatch) -> None:
    monkeypatch.setattr(selftest_module, "_project_root", lambda: None)
    console = Console(file=StringIO(), force_terminal=False)

    summary = selftest_module.run_selftest(quick=True, console=console)

    assert summary.passed
    assert summary.checked_examples == 1
