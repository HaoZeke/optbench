#!/bin/bash
# Build GMIN from the Wales svn.tar.bz2 and the OptBench LJ hook.
# The pin is the tarball sha256. The tree is not a git revision.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TARBALL="${1:-/tmp/wales-src/svn.tar.bz2}"
DEST="${2:-/tmp/wales-src}"
BUILD="${3:-/tmp/gmin-build}"
EXPECTED="02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f"

if [[ ! -f "$TARBALL" ]]; then
  echo "missing tarball $TARBALL" >&2
  exit 1
fi
got="$(sha256sum "$TARBALL" | awk '{print $1}')"
if [[ "$got" != "$EXPECTED" ]]; then
  echo "tarball sha256 $got" >&2
  exit 1
fi

if [[ ! -f "$DEST/wales/GMIN/source/lj.F90" ]]; then
  tar -xjf "$TARBALL" -C "$DEST"
fi

rm -f "$DEST/wales/INTERFACES/optbench_pot.f90"
cp "$ROOT/codes/gmin/optbench_pot.F90" "$DEST/wales/INTERFACES/optbench_pot.F90"
mkdir -p "$DEST/wales/HiRE"
cp "$ROOT/codes/gmin/hire_interface.F90" "$DEST/wales/HiRE/hire_interface.F90"

python3 - "$DEST/wales/GMIN/source/lj.F90" << 'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text()
if "OPTBENCH_ACTIVE" in text:
    raise SystemExit(0)
needle = "      USE BLAS_WRAPPERS, ONLY: DSYMV_W, DDOT_W, DGEMV_W\n"
use = needle + "      USE OPTBENCH_POT, ONLY: OPTBENCH_ACTIVE, OPTBENCH_LJ\n"
old = "      REAL(KIND = REAL64) :: R6, DIST, DUMMY\n"
hook = """      REAL(KIND = REAL64) :: R6, DIST, DUMMY

      IF (OPTBENCH_ACTIVE() .AND. .NOT. STEST) THEN
         CALL OPTBENCH_LJ(NATOMS, X, V, ELJ, GTEST)
         VT(1:NATOMS) = ELJ / DBLE(NATOMS)
         RETURN
      END IF
"""
if needle not in text or old not in text:
    raise SystemExit("lj.F90 anchors missing")
text = text.replace(needle, use, 1).replace(old, hook, 1)
path.write_text(text)
PY

mkdir -p "$BUILD"
cd "$BUILD"
FC=mpif90 CC=gcc CXX=g++ cmake "$DEST/wales/GMIN/source" \
  -DWITH_MPI=ON \
  -DWITH_ALIGN=OFF \
  -DWITH_MYBLAS=OFF \
  -DWITH_MYLAPACK=OFF \
  -DCOMPILER_SWITCH=gfortran \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++
make -j"$(nproc)" GMIN
echo "built $BUILD/GMIN"
