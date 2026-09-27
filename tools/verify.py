"""Run the CPU-only checks for the released code and artefacts."""

from __future__ import annotations

import hashlib
import ast
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference_outputs"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    if env is None:
        env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def verify_python_syntax() -> None:
    files = sorted(ROOT.rglob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"PYTHON SYNTAX PASSED: {len(files)} files")


def verify_result_manifest() -> None:
    pattern = re.compile(
        r"^\| `([^`]+)` \| ([\d,]+) \| `([0-9a-f]+)` \|"
    )
    checked = 0
    for line in (ROOT / "MANIFEST.md").read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        name, size_text, expected_prefix = match.groups()
        candidates = list((ROOT / "results").rglob(name))
        if len(candidates) != 1:
            raise RuntimeError(f"manifest path is not unique: {name!r}")
        path = candidates[0]
        expected_size = int(size_text.replace(",", ""))
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"size mismatch for {path.relative_to(ROOT)}")
        if not digest(path).startswith(expected_prefix):
            raise RuntimeError(f"hash mismatch for {path.relative_to(ROOT)}")
        checked += 1
    if checked == 0:
        raise RuntimeError("no result rows found in MANIFEST.md")
    print(f"RESULT MANIFEST PASSED: {checked} artefacts")


def verify_generated_outputs() -> None:
    def verify_pdf_metadata(path: Path) -> None:
        data = path.read_bytes()
        forbidden = (b"/CreationDate", b"/ModDate", b"/Author")
        present = [item.decode("ascii") for item in forbidden if item in data]
        if present:
            raise RuntimeError(
                f"identity-bearing PDF metadata in {path.name}: {present}"
            )
        if b"/Creator (freeknob)" not in data:
            raise RuntimeError(f"expected PDF creator missing from {path.name}")

    with tempfile.TemporaryDirectory(prefix="freeknob-verify-") as directory:
        output = Path(directory)
        env = os.environ.copy()
        env["PAPER_DIR"] = str(output)
        run([sys.executable, "paper/make_numbers.py"], env=env)
        run([sys.executable, "paper/make_figures.py"], env=env)

        expected_tex = sorted(path.name for path in REFERENCE.glob("*.tex"))
        generated_tex = sorted(path.name for path in output.glob("*.tex"))
        if generated_tex != expected_tex:
            raise RuntimeError(
                f"generated LaTeX set differs: {generated_tex} != {expected_tex}"
            )
        for name in expected_tex:
            # Python writes CRLF on Windows and LF elsewhere, so line endings
            # are normalised before the byte comparison.
            generated = (output / name).read_bytes().replace(b"\r\n", b"\n")
            reference = (REFERENCE / name).read_bytes().replace(b"\r\n", b"\n")
            if generated != reference:
                raise RuntimeError(f"generated numerical output differs: {name}")
        for name in ("fig_illustration.pdf", "fig_mechanism.pdf",
                     "fig_decomposition.pdf", "fig_contrasts.pdf"):
            path = output / name
            if not path.exists() or path.stat().st_size < 1000:
                raise RuntimeError(f"figure was not generated: {name}")
            verify_pdf_metadata(path)
            verify_pdf_metadata(REFERENCE / name)
        print(
            f"GENERATED OUTPUTS PASSED: {len(expected_tex)} exact LaTeX files, "
            "4 figures with reproducible metadata"
        )


def main() -> int:
    verify_python_syntax()
    run([
        sys.executable, "-m", "unittest", "discover", "-s", "tests",
        "-p", "test_*.py",
    ])
    verify_result_manifest()
    verify_generated_outputs()
    parity = __import__("json").loads(
        (ROOT / "results/freeknob_parity_cascast_det.json").read_text(
            encoding="utf-8"
        )
    )
    if not parity["passed"] or parity["failure_count"] != 0:
        raise RuntimeError("shipped real-data parity record does not pass")
    print("SEVIR PARITY RECORD PASSED: 30/30 cells at 1e-7")
    print("VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
