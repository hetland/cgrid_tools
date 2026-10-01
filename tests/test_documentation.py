"""Keep the short, self-contained user-facing Python examples runnable."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ("README.md", "docs/quickstart.md", "docs/metrics.md", "docs/crdt_notes.md")
SNIPPETS = [
    pytest.param(code, id=f"{page}:{index + 1}")
    for page in PAGES
    for index, code in enumerate(
        re.findall(r"```python\n(.*?)```", (ROOT / page).read_text(), re.DOTALL)
    )
]


@pytest.mark.parametrize("code", SNIPPETS)
def test_documented_python_example(code, tmp_path, monkeypatch):
    if "target_crs=" in code:
        pytest.importorskip("pyproj")
    monkeypatch.chdir(tmp_path)
    exec(compile(code, "<documentation example>", "exec"), {})
