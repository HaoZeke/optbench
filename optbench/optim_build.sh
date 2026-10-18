#!/bin/bash
# Build Wales-group OPTIM from the public svn export and link USERPOT
# to the optbench potential server. The pin is the tarball sha256.
set -eu
SHA256="02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f"
VERSION="02aaf45245af"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARBALL="${OPTIM_TARBALL:-/tmp/wales/svn.tar.bz2}"
SRC="${OPTIM_SRC:-/tmp/wales/wales}"
BUILD="${OPTIM_BUILD:-/tmp/wales/wales/OPTIM/build/gfortran}"
DEST="${OPTIM_PREFIX:-/tmp/optbench-optim}"

if [[ ! -f "$TARBALL" ]]; then
  mkdir -p "$(dirname "$TARBALL")"
  curl -L "http://www-wales.ch.cam.ac.uk/svn.tar.bz2" -o "$TARBALL"
fi
echo "$SHA256  $TARBALL" | sha256sum -c -

if [[ ! -f "$SRC/OPTIM/source/CMakeLists.txt" ]]; then
  mkdir -p "$(dirname "$SRC")"
  tar -xjf "$TARBALL" -C "$(dirname "$SRC")"
fi
printf '%s\n' "$VERSION" > "$SRC/OPTIM/source/VERSION"
cp "$ROOT/optbench/optim_userpot.f90" "$SRC/OPTIM/source/dummy_userpot.f90"

mkdir -p "$BUILD"
# /usr/bin/c++ is clang++, and this image has no GCC 14 libstdc++.
# g++ finds the GCC 13 library. OPTIM enables C++ for the build.
if [[ ! -f "$BUILD/CMakeCache.txt" ]]; then
  cmake -S "$SRC/OPTIM/source" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DCMAKE_Fortran_FLAGS="-fallow-argument-mismatch"
fi
printf '%s\n' "$VERSION" > "$BUILD/VERSION"
cmake --build "$BUILD" --target OPTIM -j "${OPTIM_JOBS:-8}"
mkdir -p "$DEST"
cp "$BUILD/OPTIM" "$DEST/OPTIM"
chmod 755 "$DEST/OPTIM"
if ! grep -q "OPTIM version ${VERSION}," "$BUILD/header.f90"; then
  echo "OPTIM version ${VERSION} is not in header.f90" >&2
  exit 1
fi
echo "built $DEST/OPTIM"
