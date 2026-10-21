#!/bin/sh
# One SoftSaddle batch on the Pt Morse island.
# The archived score is ten repetitions and up to 10500 saddle points.
# This run asks for one batch of two searches and stops after ten minutes.
# Two searches are the smallest set pdist can measure.
set -eu

root=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
upstream=$root/upstream
libdir=$root/lib
work=$root/work
mkdir -p "$libdir" "$work"

compile_lib() {
    srcdir=$1
    soname=$2
    (
        cd "$srcdir"
        # The C files are GSL sources. g++ treats them as C++ and rejects
        # size_t. Compile them as C, then link with the C++ objects.
        for src in *.cpp; do
            g++ -O3 -fPIC -std=c++14 -DBOOST_BIND_GLOBAL_PLACEHOLDERS -c "$src"
        done
        # qrstep.c is included by tridiag.c. It is not a standalone file.
        for src in tridiag.c tridiagv.c; do
            gcc -O3 -fPIC -std=c11 -c "$src"
        done
        g++ -O3 -shared -o "$libdir/$soname" ./*.o -lgsl -lgslcblas
        rm -f ./*.o
    )
}

compile_lib "$upstream/libraries/davidsson_lib/to_compile" libdavidson.so
compile_lib "$upstream/libraries/lanczos_lib/to_compile" liblanczos.so
(
    cd "$upstream/libraries/libmorse"
    g++ -O3 -shared -fPIC -std=c++14 -o "$libdir/libmorse.so" ./*.cpp -lgsl -lgslcblas
)

cp "$upstream/test_run/recorded/adaptive_HSphere-lbfgs-davidson-sm50/iniConfig.m" "$work/iniConfig.m"
cp "$upstream/source_code/displace_bech.mat" "$work/displace_bech.mat"
cp "$upstream/source_code/matlab.h" "$work/matlab.h"
cp "$upstream/source_code/morse.h" "$work/morse.h"

sed -i \
    -e 's/do_repetition = true;/do_repetition = false;/' \
    -e 's/desired_number_sp = 10500;/desired_number_sp = 1;/' \
    -e 's/parall_search_per_min=25;/parall_search_per_min=2;/' \
    -e 's/max_attempts = 220;/max_attempts = 1;/' \
    -e 's/real_n_cores=25;/real_n_cores=2;/' \
    -e 's/RUNTIME = 86400\*10;/RUNTIME = 600;/' \
    -e 's/read_from_database = true;/read_from_database = false;/' \
    -e 's/n_repetition = 10;/n_repetition = 1;/' \
    -e "s|lib_path=.*;|lib_path='$libdir';|" \
    "$work/iniConfig.m"

export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "libraries in $libdir"
ls -l "$libdir"
echo "smoke config written"
grep -E "do_repetition|desired_number_sp|max_attempts|RUNTIME|read_from_database|n_repetition|real_n_cores|parall_search" "$work/iniConfig.m"
