"""
Execute the Python code blocks in ``README.md`` and ``docs/*.md``.

Guards the prose against the API drifting underneath it: an earlier pair of
GPU guides was never run and still unpacked a return value that had been
removed (issue #56). The blocks of one document run in order in a shared
namespace, as a reader who pastes them one after another would see them.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.behaviour

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
FENCE = re.compile(r"^```python[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)


def python_blocks(path):
    return [match.group(1) for match in FENCE.finditer(path.read_text())]


def test_docs_contain_python_blocks():
    assert DOCS, "no documents found under docs/"
    assert any(python_blocks(path) for path in DOCS)


@pytest.mark.parametrize("path", DOCS, ids=lambda path: path.name)
def test_docs_code_blocks_run(path, capsys):
    namespace = {"__name__": "__docs__"}
    for index, source in enumerate(python_blocks(path), start=1):
        try:
            exec(compile(source, f"{path.name}:block{index}", "exec"), namespace)
        except Exception as error:  # pragma: no cover - the message is the point
            captured = capsys.readouterr()
            pytest.fail(
                f"{path.name}, python block {index} raised "
                f"{type(error).__name__}: {error}\n"
                f"--- block ---\n{source}\n--- stdout so far ---\n{captured.out}"
            )
