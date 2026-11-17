"""Runs every script in examples/ and checks the results they print."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# For each script, lines that must appear in its output.
EXPECTED = {
    "shortest_path.py": [
        "cost:   730.0",
        "route:  Lisbon -> Madrid -> Lyon -> Zurich -> Berlin",
        "Paris to Zurich: unreachable",
        "still acyclic? False",
    ],
    "resource_limits.py": [
        "  route: Lisbon -> Porto -> Bordeaux -> Lyon -> Zurich -> Berlin",
        "  cost:  770.0  uses: {'hours': 34.0, 'tolls': 2.0}",
        "with only 28 hours: infeasible",
    ],
    "vessel_routing.py": [
        "Best voyage with unlimited fuel: profit 127, 24 days at sea",
        "Best voyage on one tank (20 days at sea): profit 85, 18 days at sea",
    ],
    "column_generation.py": [
        "LP optimum: 1061.00 with 10 generated routes",
        "Integer plan: 1061.00",
    ],
    "fleet_pricing.py": [
        "LP optimum: 4637.00 with 102 generated routes",
        "Integer plan from these routes: 4637.00",
    ],
}

pytestmark = pytest.mark.skipif(not EXAMPLES.is_dir(), reason="examples/ is not available")


def test_every_example_is_covered():
    assert sorted(path.name for path in EXAMPLES.glob("*.py")) == sorted(EXPECTED)


@pytest.mark.parametrize("script", sorted(EXPECTED))
def test_example(script):
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / script)], capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    for expected in EXPECTED[script]:
        assert expected in lines, f"{script} did not print {expected!r}"
