# Contributing

Bug reports, questions and pull requests are welcome on
[GitHub](https://github.com/sharabhshukla/CelerisRoute).

## Development setup

You need a C++20 compiler and the Boost headers (see the
[installation guide](https://sharabhshukla.github.io/CelerisRoute/getting-started/installation/)).
oneTBB is optional.

```bash
git clone https://github.com/sharabhshukla/CelerisRoute
cd CelerisRoute
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

The editable install compiles the extension once. After changing C++ code, run
`pip install -e ".[dev]"` again; Python changes need no rebuild.

## Where things are

```text
cpp/include/celerisroute/  C++ headers of the solver core
cpp/src/                   C++ sources of the solver core
cpp/bindings/python.cpp    nanobind module celerisroute._core
cpp/bench/                 native benchmark harness
src/celerisroute/          the Python package
tests/                     pytest suite
examples/                  runnable scripts, also executed by the tests
benchmarks/                benchmark scripts and their recorded results
docs/                      MkDocs site
scripts/                   helper scripts for CI
```

## Tests

```bash
python -m pytest                  # everything
python -m pytest tests/test_rcsp.py -k tolerance
```

Solvers are tested against brute-force path enumeration in
`tests/_reference.py`. A change to a solver should come with a test that
compares it with that reference on random graphs.

The examples and the code blocks of the documentation run as part of the test
suite, so they cannot go stale.

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve                      # live preview on http://127.0.0.1:8000
mkdocs build --strict             # what CI runs
```

Docstrings use the NumPy style and are rendered into the API reference.

## Benchmarks

Benchmark numbers in the documentation come from the JSON files in
`benchmarks/results/`. If you change something that affects performance,
re-run the relevant script and say in the pull request what machine it ran on.
Please do not edit numbers by hand.

## Pull requests

- Keep a pull request to one topic.
- Add a line to `CHANGELOG.md`.
- CI builds and tests on Linux, macOS and Windows; all jobs should pass.
