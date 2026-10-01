"""The browser's bundle must be readable by the Python that unpacks it.

The input builder writes the structure and the manifest into a ZIP by hand, so
that the page needs no compression library. The archive is useless unless the
Colab side can open it, which is `zipfile` -- so the JavaScript is executed in
Node here and the bytes it produces are handed straight to `zipfile`.
"""

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest


WEB = Path("web")
ZIP_JS = WEB / "zip.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required to run the browser ZIP writer"
)


def build_zip(files: list[tuple[str, str]], tmp_path: Path) -> Path:
    """Run the page's ZIP writer in Node and return the archive it produced."""

    output = tmp_path / "bundle.zip"
    script = tmp_path / "run.mjs"
    script.write_text(
        "const { createRequire } = await import('node:module');\n"
        "const require = createRequire(process.argv[1]);\n"
        f"const {{ storedZipBytes }} = require({str(ZIP_JS.resolve())!r});\n"
        f"const files = {json.dumps(files)};\n"
        "const bytes = storedZipBytes(files, new Date('2026-10-01T12:00:00Z'));\n"
        "const fs = require('node:fs');\n"
        f"fs.writeFileSync({str(output)!r}, Buffer.from(bytes));\n",
        encoding="utf-8",
    )
    subprocess.run(["node", str(script)], check=True, capture_output=True, text=True)
    return output


def test_bundle_opens_and_round_trips_its_contents(tmp_path):
    files = [
        ("protein.pdb", "ATOM      1  N   ALA A   1      1.000   2.000   3.000\nEND\n"),
        ("system.yaml", "project:\n  name: demo\n"),
        ("README.txt", "mdprep input bundle\n"),
    ]

    archive = build_zip(files, tmp_path)

    with zipfile.ZipFile(archive) as zf:
        assert zf.testzip() is None  # no CRC mismatches
        assert zf.namelist() == [name for name, _ in files]
        for name, text in files:
            assert zf.read(name).decode("utf-8") == text


def test_bundle_extracts_to_disk(tmp_path):
    """Exactly what the Colab cell does: ZipFile(...).extractall('.')."""
    files = [("system.yaml", "project:\n  name: demo\n"), ("x.pdb", "END\n")]
    archive = build_zip(files, tmp_path)
    target = tmp_path / "out"
    target.mkdir()

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)

    assert (target / "system.yaml").read_text(encoding="utf-8").startswith("project:")
    assert (target / "x.pdb").exists()


def test_bundle_handles_non_ascii_and_empty_content(tmp_path):
    files = [("notes.txt", "ångström – Å\n"), ("empty.txt", "")]

    archive = build_zip(files, tmp_path)

    with zipfile.ZipFile(archive) as zf:
        assert zf.testzip() is None
        assert zf.read("notes.txt").decode("utf-8") == "ångström – Å\n"
        assert zf.read("empty.txt") == b""


def test_bundle_survives_a_realistic_structure(tmp_path):
    """A real PDB is large enough to catch offset arithmetic mistakes."""
    source = Path("tests/data/protein_with_waters.pdb").read_text(encoding="utf-8")
    files = [("protein_with_waters.pdb", source), ("system.yaml", "project:\n")]

    archive = build_zip(files, tmp_path)

    with zipfile.ZipFile(archive) as zf:
        assert zf.testzip() is None
        assert zf.read("protein_with_waters.pdb").decode("utf-8") == source


def test_crc32_matches_the_python_implementation(tmp_path):
    """A wrong CRC makes an archive that opens but fails verification."""
    import binascii

    script = tmp_path / "crc.mjs"
    script.write_text(
        "const { createRequire } = await import('node:module');\n"
        "const require = createRequire(process.argv[1]);\n"
        f"const {{ crc32 }} = require({str(ZIP_JS.resolve())!r});\n"
        "const data = new TextEncoder().encode('mdprep bundle contents');\n"
        "console.log(crc32(data));\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["node", str(script)], check=True, capture_output=True, text=True
    )

    expected = binascii.crc32(b"mdprep bundle contents") & 0xFFFFFFFF
    assert int(result.stdout.strip()) == expected
