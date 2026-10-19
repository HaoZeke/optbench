#!/usr/bin/env bash
# Build the current Wales GMIN from svn.tar.bz2. The source is unmodified.
# The pin is the sha256 of the tarball fetched 2026-10-19.
# SVNREV inside the tree is df8da599267534fbe43a250e578f7396a7946311.
set -euo pipefail

sha=738fafe3df5ad73100b7e7d7bbbc407ee8cd5d969c2c0a58f2c109167f463d31
tarball=${GMIN_TARBALL:-/tmp/wales-latest/svn.tar.bz2}
src=${GMIN_SRC:-/tmp/wales-latest/src}
build=${GMIN_BUILD:-/tmp/gmin-latest}

if [[ ! -f "$tarball" ]]; then
  mkdir -p "$(dirname "$tarball")"
  curl -fsSL -o "$tarball" "https://www-wales.ch.cam.ac.uk/svn.tar.bz2"
fi
echo "$sha  $tarball" | sha256sum -c -

if [[ ! -f "$src/wales/GMIN/source/lj.f90" && ! -f "$src/wales/GMIN/source/lj.F90" ]]; then
  mkdir -p "$src"
  tar -xjf "$tarball" -C "$src"
fi

mkdir -p "$build"
CC=gcc CXX=g++ FC=gfortran cmake -S "$src/wales/GMIN/source" -B "$build" \
  -DWITH_MPI=OFF \
  -DCOMPILER_SWITCH=gfortran \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --target GMIN -j"$(nproc)"
echo "built $build/GMIN"
