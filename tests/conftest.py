"""
Collection-time check: every test says what kind of evidence it is.

``AGENTS.md`` ("Changing the math") and ``CONTRIBUTING.md`` distinguish tests
that verify the mathematics against an independent reference from tests that
pin the library's current behaviour. The two markers make that distinction
queryable (``pytest -m math`` runs only the proofs) and this hook keeps it
complete: a test carrying neither marker, or both, fails collection.

The hook runs after pytest's own ``-k``/``-m`` deselection (``trylast``), so
it judges only the tests that were actually selected: a scoped run is never
aborted by an untagged test elsewhere in the tree. A full run, which CI
always does, checks every test.
"""

import pytest

KIND_MARKERS = ("math", "behaviour")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    untagged = []
    double = []
    for item in items:
        kinds = {marker.name for marker in item.iter_markers()} & set(KIND_MARKERS)
        if not kinds:
            untagged.append(item.nodeid)
        elif len(kinds) > 1:
            double.append(item.nodeid)
    problems = []
    if untagged:
        problems.append(
            "tests without a kind marker (add @pytest.mark.math or "
            "@pytest.mark.behaviour, or a module-level pytestmark):\n  "
            + "\n  ".join(untagged)
        )
    if double:
        problems.append(
            "tests marked both math and behaviour (pick one):\n  " + "\n  ".join(double)
        )
    if problems:
        raise pytest.UsageError("\n".join(problems))
