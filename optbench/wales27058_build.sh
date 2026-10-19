#!/bin/bash
# Build the OPTIM target from optbench.org/codes/wales-27058.tar.bz2.
# The public Wales svn.tar.bz2 snapshot is a later tree. Its direct NEWNEB
# path stores the archived RMS tolerance in RMSTOLNN and stops on RMSTOL.
set -euo pipefail

url="https://optbench.org/codes/wales-27058.tar.bz2"
sha="144a703eef674a826191308b9f8dfda548d88fc7a50323687e7cc1f6e64bf348"
root="${OPTIM_27058_ROOT:-/tmp/wales27058}"
mkdir -p "$root"
cd "$root"
if [[ ! -f wales-27058.tar.bz2 ]]; then
  curl -fL --retry 3 -o wales-27058.tar.bz2 "$url"
fi
echo "$sha  wales-27058.tar.bz2" | sha256sum -c -
if [[ ! -d wales/OPTIM/source ]]; then
  tar -xjf wales-27058.tar.bz2
fi
echo 27058 > wales/OPTIM/source/VERSION
python3 - << 'PY'
from pathlib import Path
path = Path("wales/CMakeModules/FindFORTRANCOMPILER.cmake")
text = path.read_text()
old = '-ffixed-line-length-200 -ffree-line-length-200 -fPIC'
new = old + " -std=legacy -fallow-argument-mismatch -fallow-invalid-boz"
if old not in text:
    raise SystemExit("gfortran flags line not found")
if new not in text:
    path.write_text(text.replace(old, new, 1))
PY
rm -rf build
mkdir build
cd build
CC=gcc CXX=g++ FC=gfortran cmake ../wales/OPTIM/source
make -j "${OPTIM_27058_JOBS:-8}" OPTIM
echo "OPTIM_27058_BIN=$root/build/OPTIM"
