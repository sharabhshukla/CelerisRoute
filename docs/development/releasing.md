# Releasing

Releases are built by GitHub Actions. Nothing is compiled on a developer's
machine.

## What gets built

The [Wheels workflow](https://github.com/sharabhshukla/CelerisRoute/actions/workflows/wheels.yml)
uses [cibuildwheel](https://cibuildwheel.pypa.io/) to build one wheel per
platform and Python version, and runs the full test suite against every wheel
before it is kept.

| platform | architectures |
|---|---|
| Linux (manylinux and musllinux) | x86_64, aarch64 |
| macOS 11 or newer | arm64, x86_64 |
| Windows | x86_64 |

Python 3.11 to 3.15, plus the free-threaded builds of 3.14 and 3.15.

The wheels are built **without oneTBB** and against pinned Boost headers that
`scripts/fetch_boost.py` downloads and verifies by checksum. A wheel therefore
depends on nothing but the C++ runtime of the platform. A source distribution
is built alongside for everything else.

## Making a release

1. Set the version in `pyproject.toml` and add a section to `CHANGELOG.md`.
2. Push to `main` and wait for the Tests and Wheels workflows to pass.
3. Create a GitHub release with the tag `v<version>`.

Publishing the release starts the Wheels workflow once more, and its last job
uploads the wheels and the source distribution to PyPI.

!!! note "One-time setup for PyPI"
    The upload uses [trusted publishing](https://docs.pypi.org/trusted-publishers/),
    so no token is stored in the repository. It has to be enabled once on PyPI:
    add a trusted publisher for the project `celerisroute` with owner
    `sharabhshukla`, repository `CelerisRoute`, workflow `wheels.yml` and
    environment `pypi`. Until that is done, the publish job fails and nothing
    is uploaded.

## Trying a build first

Every run of the Wheels workflow keeps its wheels as artifacts. To try them
before a release, download them and upload to TestPyPI:

```bash
gh run download <run-id> --pattern "dist-*" --dir dist     # one folder per artifact
uv publish --publish-url https://test.pypi.org/legacy/ dist/*/*
```

A file name can be uploaded to an index only once, so a second attempt needs a
new version number.
