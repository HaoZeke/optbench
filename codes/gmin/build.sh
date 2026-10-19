#!/usr/bin/env bash
# Build libgminlbfgs.so from the pinned Wales svn.tar.bz2.
# The objects are source/mymylbfgs.F90 and source/blas_wrappers.F90.
# stubs.f90 and driver.f90 are the OptBench callback.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
sha=02aaf45245af7b521b5792fa77ffa364f67d756327ea9e4ecf580c4b6f83e24f
rev=df8da599267534fbe43a250e578f7396a7946311
prefix=${PREFIX:-/opt/gmin}
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
  ./wales/GMIN/source/mymylbfgs.F90 \
  ./wales/GMIN/source/blas_wrappers.F90 \
  ./wales/GMIN/source/SVNREV
got=$(tr -d '[:space:]' < "$work/wales/GMIN/source/SVNREV")
if [[ "$got" != "$rev" ]]; then
  echo "SVNREV is $got, expected $rev" >&2
  exit 1
fi

cp "$here/stubs.f90" "$here/driver.f90" "$work/"
flags=(-fPIC -O2 -fmax-stack-var-size=32768 -ffree-line-length-none)
(
  cd "$work"
  gfortran -c "${flags[@]}" -std=f2008 stubs.f90
  gfortran -c "${flags[@]}" wales/GMIN/source/blas_wrappers.F90
  gfortran -c "${flags[@]}" wales/GMIN/source/mymylbfgs.F90
  gfortran -c "${flags[@]}" driver.f90
  gfortran -shared -Wl,--no-undefined -o libgminlbfgs.so \
    stubs.o blas_wrappers.o mymylbfgs.o driver.o -lblas
)
cp "$work/libgminlbfgs.so" "$here/libgminlbfgs.so"
if mkdir -p "$prefix/lib" 2>/dev/null && cp "$here/libgminlbfgs.so" "$prefix/lib/libgminlbfgs.so"; then
  echo "installed $prefix/lib/libgminlbfgs.so"
else
  sudo mkdir -p "$prefix/lib"
  sudo cp "$here/libgminlbfgs.so" "$prefix/lib/libgminlbfgs.so"
  echo "installed $prefix/lib/libgminlbfgs.so"
fi
