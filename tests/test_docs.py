"""Executes the ``python`` code blocks of the README and the documentation.

The blocks of one page run in order and share a namespace, like cells of a
notebook. Illustrative fragments that are not meant to run are fenced as
``py`` and skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted([ROOT / "README.md", *(ROOT / "docs").rglob("*.md")])
BLOCK = re.compile(r"^```python[^\n]*\n(.*?)^```$", re.MULTILINE | re.DOTALL)


def code_blocks(page: Path) -> list[str]:
    # Blocks that only include another file (the example scripts) are run by
    # test_examples.py.
    return [block for block in BLOCK.findall(page.read_text()) if "--8<--" not in block]


RUNNABLE = [page for page in PAGES if page.is_file() and code_blocks(page)]


@pytest.mark.skipif(not RUNNABLE, reason="README and docs/ are not available")
def test_pages_were_found():
    assert ROOT / "README.md" in RUNNABLE


@pytest.mark.parametrize("page", RUNNABLE, ids=lambda page: str(page.relative_to(ROOT)))
def test_code_blocks_run(page, capsys):
    namespace: dict[str, object] = {"__name__": "__docs__"}
    for number, block in enumerate(code_blocks(page), start=1):
        exec(compile(block, f"{page.name} (block {number})", "exec"), namespace)
