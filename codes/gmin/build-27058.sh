#!/bin/bash
# Build GMIN from the 2014 OptBench tarball wales-27058.tar.bz2.
# The pin is the tarball sha256. The tree is not a git revision.
# The LJ38 CSM binary was gfortran 13.3.0, cmake Release, WITH_MPI off,
# system BLAS and LAPACK. The compile line is
# -ffixed-line-length-200 -ffree-line-length-200 -fPIC -O3 -DNDEBUG.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TARBALL="${1:-/tmp/wales-27058/wales-27058.tar.bz2}"
DEST="${2:-/tmp/wales-27058}"
BUILD="${3:-/tmp/gmin-build}"
URL="https://theory.cm.utexas.edu/benchmarks/codes/wales-27058.tar.bz2"
EXPECTED="144a703eef674a826191308b9f8dfda548d88fc7a50323687e7cc1f6e64bf348"

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

python3 - "$DEST/wales/GMIN/source/biased_torsional_sampling.f90" << 'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text()
old = "C(3*(a3-1)+1:3*(a3-1)+1) - C(3*(a2-1)+1:3*(a2-1)+1)"
new = "C(3*(a3-1)+1:3*(a3-1)+3) - C(3*(a2-1)+1:3*(a2-1)+3)"
old2 = "C(3*(a4-1)+1:3*(a4-1)+1) - C(3*(a3-1)+1:3*(a3-1)+1)"
new2 = "C(3*(a4-1)+1:3*(a4-1)+3) - C(3*(a3-1)+1:3*(a3-1)+3)"
if old not in text or old2 not in text:
    raise SystemExit(0)
path.write_text(text.replace(old, new, 1).replace(old2, new2, 1))
PY

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
