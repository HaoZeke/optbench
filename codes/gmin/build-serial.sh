#!/bin/bash
# Serial GMIN from the Wales svn tarball. No MPI and no potential hook:
# the archived data file runs as GMIN's own Lennard-Jones basin-hopping.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SHA256="738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31"
SVNREV="df8da599267534fbe43a250e578f7396a7946311"
URL="https://www-wales.ch.cam.ac.uk/svn.tar.bz2"
SRC="${1:-$ROOT/build/wales-src}"
BUILD="${2:-$ROOT/build/gmin-serial}"

mkdir -p "$SRC" "$BUILD"
tarball="$SRC/svn.tar.bz2"
if [[ ! -f "$tarball" ]]; then
  curl -fsSL -o "$tarball" "$URL"
fi
echo "$SHA256  $tarball" | sha256sum -c -
tar -xjf "$tarball" -C "$SRC" ./wales/GMIN ./wales/CMakeModules ./wales/INTERFACES ./wales/SCRIPTS ./wales/HiRE
got="$(tr -d '[:space:]' < "$SRC/wales/GMIN/source/SVNREV")"
if [[ "$got" != "$SVNREV" ]]; then
  echo "SVNREV is $got, expected $SVNREV" >&2
  exit 1
fi

cmake -S "$SRC/wales/GMIN/source" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_MPI=OFF \
  -DWITH_ALIGN=OFF \
  -DWITH_MYBLAS=OFF \
  -DWITH_MYLAPACK=OFF \
  -DCOMPILER_SWITCH=gfortran
cmake --build "$BUILD" --target GMIN -j "$(nproc)"
echo "built $BUILD/GMIN"
