"""Download the Boost headers that celerispath needs to compile.

celerispath uses only header-only parts of Boost (Boost.Graph and what it
includes), so nothing has to be built. This script downloads one pinned Boost
release, checks its SHA-256 and unpacks the ``boost/`` header directory:

    python scripts/fetch_boost.py .deps/boost
    pip install . --config-settings=cmake.define.BOOST_ROOT=$PWD/.deps/boost

The continuous-integration builds use it where no package manager offers a
recent Boost (Windows, the manylinux containers). On a developer machine the
system package (``apt install libboost-dev``, ``brew install boost``) is the
simpler choice.
"""

import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

VERSION = "1.88.0"
SHA256 = "3621533e820dcab1e8012afd583c0c73cf0f77694952b81352bf38c1488f9cb4"
NAME = "boost_" + VERSION.replace(".", "_")
URL = f"https://archives.boost.io/release/{VERSION}/source/{NAME}.tar.gz"

destination = Path(sys.argv[1] if len(sys.argv) > 1 else ".deps/boost").resolve()
marker = destination / "boost" / "version.hpp"
if marker.exists():
    print(f"Boost headers already present in {destination}")
    sys.exit(0)

destination.mkdir(parents=True, exist_ok=True)
archive = destination / f"{NAME}.tar.gz"

print(f"downloading {URL}")
digest = hashlib.sha256()
with urllib.request.urlopen(URL) as response, open(archive, "wb") as file:
    while chunk := response.read(1 << 20):
        digest.update(chunk)
        file.write(chunk)
if digest.hexdigest() != SHA256:
    archive.unlink()
    sys.exit(f"checksum mismatch for {URL}: got {digest.hexdigest()}, expected {SHA256}")

print(f"unpacking the headers into {destination}")
prefix = f"{NAME}/boost/"
with tarfile.open(archive) as tar:
    for member in tar:
        if member.isfile() and member.name.startswith(prefix):
            member.name = member.name[len(NAME) + 1:]  # drop the leading "boost_1_88_0/"
            tar.extract(member, destination, filter="data")
archive.unlink()

if not marker.exists():
    sys.exit(f"the archive did not contain {prefix}version.hpp")
print(f"done: set BOOST_ROOT={destination}")
