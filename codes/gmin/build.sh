#!/usr/bin/env bash
# Build the 2014 GMIN binary named by optbench.org codes/wales-27058.tar.bz2.
# The pin is the sha256 of that file. SVNREV inside the tree is 27059.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
sha=144a703eef674a826191308b9f8dfda548d88fc7a50323687e7cc1f6e64bf348
tarball=${GMIN_TARBALL:-/tmp/wales-27058.tar.bz2}
src=${GMIN_SRC:-/tmp/wales-27058}
build=${GMIN_BUILD:-/tmp/gmin-build}

if [[ ! -f "$tarball" ]]; then
  curl -fsSL -o "$tarball" "https://optbench.org/codes/wales-27058.tar.bz2"
fi
echo "$sha  $tarball" | sha256sum -c -

if [[ ! -f "$src/wales/GMIN/source/lj.f" ]]; then
  mkdir -p "$src"
  tar -xjf "$tarball" -C "$src"
fi

torsion="$src/wales/GMIN/source/biased_torsional_sampling.f90"
if ! grep -q '3\*(a3-1)+3' "$torsion"; then
  patch -d "$src" -p1 < "$here/gfortran13-biased-torsion.patch"
fi

mkdir -p "$build"
CC=gcc CXX=g++ FC=gfortran cmake -S "$src/wales/GMIN/source" -B "$build" \
  -DWITH_MPI=OFF \
  -DCOMPILER_SWITCH=gfortran \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --target GMIN -j"$(nproc)"
echo "built $build/GMIN"
