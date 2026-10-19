#!/bin/bash
# Build current GMIN, SVNREV df8da599267534fbe43a250e578f7396a7946311.
# The pin is the sha256 of https://www-wales.ch.cam.ac.uk/svn.tar.bz2 fetched 2026-10-19.
# Serial gfortran 13.3.0, cmake Release, WITH_MPI off, system BLAS and LAPACK.
# The GMIN compile line is
# -ffixed-line-length-200 -ffree-line-length-200 -fPIC -fallow-argument-mismatch -std=legacy -ffixed-line-length-132 -ffree-line-length-0 -O3
set -euo pipefail

TARBALL="${1:-/tmp/wales-src/svn.tar.bz2}"
DEST="${2:-/tmp/wales-src/tree}"
BUILD="${3:-/tmp/gmin-df8da599}"
URL="https://www-wales.ch.cam.ac.uk/svn.tar.bz2"
EXPECTED="738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31"
REV="df8da599267534fbe43a250e578f7396a7946311"

if [[ ! -f "$TARBALL" ]]; then
  mkdir -p "$(dirname "$TARBALL")"
  curl -fL --retry 3 -o "$TARBALL" "$URL"
fi
got="$(sha256sum "$TARBALL" | awk '{print $1}')"
if [[ "$got" != "$EXPECTED" ]]; then
  echo "tarball sha256 $got" >&2
  exit 1
fi

if [[ ! -f "$DEST/wales/GMIN/source/CMakeLists.txt" ]]; then
  mkdir -p "$DEST"
  tar -xjf "$TARBALL" -C "$DEST"
fi
rev="$(tr -d '[:space:]' < "$DEST/wales/GMIN/source/SVNREV")"
if [[ "$rev" != "$REV" ]]; then
  echo "SVNREV is $rev" >&2
  exit 1
fi

mkdir -p "$BUILD"
cd "$BUILD"
FC=gfortran cmake "$DEST/wales/GMIN/source" \
  -DWITH_MPI=OFF \
  -DWITH_MYBLAS=OFF \
  -DWITH_MYLAPACK=OFF \
  -DCOMPILER_SWITCH=gfortran \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_Fortran_FLAGS="-ffixed-line-length-200 -ffree-line-length-200 -fPIC -fallow-argument-mismatch -std=legacy"
make -j"$(nproc)" GMIN
echo "built $BUILD/GMIN"
