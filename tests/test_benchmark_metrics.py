"""
The TCPDBench metrics used by ``benchmarks/tcpd.py`` reproduce the values
published with them: the doctests in ``benchmarks/metrics.py`` are TCPDBench's
own examples (van den Burg and Williams, 2020), computed by their code.
"""

import doctest
import importlib.util
import os

import pytest

pytestmark = pytest.mark.math

METRICS = os.path.join(os.path.dirname(__file__), os.pardir, "benchmarks", "metrics.py")


def test_metrics_match_tcpdbench_examples():
    spec = importlib.util.spec_from_file_location("benchmark_metrics", METRICS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = doctest.testmod(module, verbose=False)
    assert result.attempted >= 20
    assert result.failed == 0
