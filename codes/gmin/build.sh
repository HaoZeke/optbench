#!/usr/bin/env bash
# Build the GMIN executable from the Wales group's public svn.tar.bz2.
# The pin is the tarball sha256 and the SVNREV string inside that tarball.
# harness_pot.f90 and potential.patch route POTENTIAL through a callback
# when one is registered, and can trace native calls for the matched run.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
sha=738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31
rev=df8da599267534fbe43a250e578f7396a7946311
tarball=${GMIN_TARBALL:-}

if [[ -z "$tarball" ]]; then
  if [[ -f /tmp/gmin-src/svn.tar.bz2 ]]; then
    tarball=/tmp/gmin-src/svn.tar.bz2
  else
    tarball=$here/svn.tar.bz2
    curl -fsSL -o "$tarball" "https://www-wales.ch.cam.ac.uk/svn.tar.bz2"
  fi
fi

echo "$sha  $tarball" | sha256sum -c -

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
tar -xjf "$tarball" -C "$work" \
  ./wales/GMIN \
  ./wales/CMakeModules \
  ./wales/INTERFACES \
  ./wales/SCRIPTS \
  ./wales/HiRE
got=$(tr -d '[:space:]' < "$work/wales/GMIN/source/SVNREV")
if [[ "$got" != "$rev" ]]; then
  echo "SVNREV is $got, expected $rev" >&2
  exit 1
fi

cp "$here/harness_pot.f90" "$work/wales/GMIN/source/harness_pot.f90"
patch -p1 -d "$work/wales/GMIN/source" < "$here/potential.patch"

cmake -S "$work/wales/GMIN/source" -B "$work/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOMPILER_SWITCH=gfortran \
  -DWITH_ALIGN=OFF \
  -DWITH_MPI=OFF \
  -DWITH_SPARSE=OFF \
  -DWITH_MYBLAS=OFF \
  -DWITH_MYLAPACK=OFF
cmake --build "$work/build" --target GMIN -j "$(nproc)"

mkdir -p "$here/build"
cp "$work/build/GMIN" "$here/build/GMIN"
prefix=${PREFIX:-/opt/gmin}
if mkdir -p "$prefix/bin" 2>/dev/null && cp "$here/build/GMIN" "$prefix/bin/GMIN"; then
  echo "installed $prefix/bin/GMIN"
else
  sudo mkdir -p "$prefix/bin"
  sudo cp "$here/build/GMIN" "$prefix/bin/GMIN"
  echo "installed $prefix/bin/GMIN"
fi
