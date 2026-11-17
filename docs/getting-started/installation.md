# Installation

## From a wheel

CelerisPath is in a pre-release phase. Wheels are published on TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ celerispath
```

The second index is where pip finds NumPy and SciPy, the only run-time
dependencies. A wheel needs no compiler, no Boost and no other library.

Then check that it works:

```python
import celerispath as cp

print(cp.__version__)
```

!!! note "Wheels and oneTBB"
    Release wheels are built without oneTBB, so they depend on nothing but the
    system C++ runtime. `PricingFleet` then schedules its work on plain native
    threads, which measured as fast as oneTBB in the
    [fleet benchmark](../performance.md#fleet-pricing). `cp.tbb_available()`
    tells you which build you have.

## From source

Building needs a C++20 compiler and the Boost headers. Everything else is
fetched by pip.

| | |
|---|---|
| Python | 3.11 or newer, free-threaded builds included |
| C++ compiler | C++20: a recent GCC, Clang, Apple Clang or MSVC 2022 |
| CMake | 3.21 or newer (pip fetches it if missing) |
| Boost | headers only, 1.74 or newer; nothing is linked |
| oneTBB | optional, 2021 or newer |

=== "Debian / Ubuntu"

    ```bash
    sudo apt install libboost-dev        # required
    sudo apt install libtbb-dev          # optional
    ```

=== "Fedora"

    ```bash
    sudo dnf install boost-devel         # required
    sudo dnf install tbb-devel           # optional
    ```

=== "macOS"

    ```bash
    brew install boost                   # required
    brew install tbb                     # optional
    ```

=== "conda"

    ```bash
    conda install -c conda-forge libboost-headers   # required
    conda install -c conda-forge tbb-devel          # optional
    ```

=== "Windows"

    ```powershell
    # Downloads a pinned Boost release and unpacks only the headers.
    python scripts/fetch_boost.py .deps/boost
    $env:BOOST_ROOT = "$PWD\.deps\boost"
    ```

Then install from a checkout, or straight from GitHub:

```bash
pip install .                                   # from a checkout
pip install -e ".[dev]"                         # editable, with pytest
pip install git+https://github.com/sharabhshukla/CelerisPath
```

### Build options

No path is hardcoded in the build. If Boost or oneTBB live somewhere unusual,
say where:

```bash
pip install . --config-settings=cmake.define.BOOST_ROOT=/path/to/prefix
pip install . --config-settings=cmake.define.TBB_DIR=/path/to/prefix/lib/cmake/TBB
```

`CELERISPATH_WITH_TBB` controls the oneTBB dependency:

| value | meaning |
|---|---|
| `AUTO` (default) | use oneTBB when CMake finds it |
| `ON` | fail the build when oneTBB is missing |
| `OFF` | never link oneTBB |

```bash
pip install . --config-settings=cmake.define.CELERISPATH_WITH_TBB=OFF
```

oneTBB is linked as a shared library, so an extension built with it needs
`libtbb` at run time.

## Supported platforms

Every commit is built and tested on Linux, macOS and Windows; see the
[build status](https://github.com/sharabhshukla/CelerisPath/actions) and
[Releasing](../development/releasing.md) for the list of wheels.
