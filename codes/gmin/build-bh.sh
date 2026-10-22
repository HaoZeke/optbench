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

# A replica that hits the target inside SYMMETRY used to GOTO 37 on its own.
# The other ranks then block forever in BHPT_EXCHANGE. The same split happens
# when one rank reaches MAX_NPCALL. Leave through BHPT_TARGET when NPAR > 1.
python3 - "$DEST/wales/GMIN/source/mc.F90" << 'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text()
if "One rank leaving on HIT deadlocks" in text:
    raise SystemExit(0)
old1 = """! check we actually want to enter the basin-hopping loop
   IF (HIT.OR.(NSTEPS.EQ.0)) GOTO 37
"""
new1 = """! check we actually want to enter the basin-hopping loop
#ifdef GMIN_MPI
   ! One rank leaving on HIT deadlocks the others in BHPT_EXCHANGE.
   IF ((HIT.AND.NPAR.LE.1).OR.(NSTEPS.EQ.0)) GOTO 37
#else
   IF (HIT.OR.(NSTEPS.EQ.0)) GOTO 37
#endif
"""
old2 = """      IF (MAX_NPCALLT .AND. (NPCALL .GE. MAX_NPCALL)) THEN
         write(myunit,"(A,I18,A,I18,A)") "mc> MAX_NPCALL: maximum number of potential calls reached ( ", &
     &           npcall, " > ", max_npcall, " ).  Ending MC loop"
         GOTO 37
      ENDIF
"""
new2 = """      IF (MAX_NPCALLT .AND. (NPCALL .GE. MAX_NPCALL)) THEN
         write(myunit,"(A,I18,A,I18,A)") "mc> MAX_NPCALL: maximum number of potential calls reached ( ", &
     &           npcall, " > ", max_npcall, " ).  Ending MC loop"
#ifdef GMIN_MPI
         IF (NPAR.GT.1) THEN
            HIT=.TRUE.
         ELSE
            GOTO 37
         ENDIF
#else
         GOTO 37
#endif
      ENDIF
"""
old3 = """            IF (HIT) GOTO 37 
            ! ELSE
"""
new3 = """#ifdef GMIN_MPI
            IF (HIT.AND.NPAR.LE.1) GOTO 37
#else
            IF (HIT) GOTO 37 
#endif
            ! ELSE
"""
for name, old, new in (("loop", old1, new1), ("cap", old2, new2), ("sym", old3, new3)):
    if text.count(old) != 1:
        raise SystemExit(f"mc.F90 anchor {name} count is {text.count(old)}")
    text = text.replace(old, new, 1)
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
