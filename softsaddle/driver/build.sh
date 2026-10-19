#!/bin/sh
# Build the Lanczos and Davidson shared libraries used by the harness.
# Morse.cpp and matlab.cpp stay out: the potential is the spec surface.
set -eu

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
libdir=$root/lib
driver=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
mkdir -p "$libdir"

compile_lib() {
    name=$1
    srcdir=$2
    define=$3
    objdir=$libdir/obj-$name
    rm -rf "$objdir"
    mkdir -p "$objdir"
    for src in "$srcdir"/*.cpp; do
        base=$(basename "$src")
        case $base in
            Morse.cpp|matlab.cpp) continue ;;
        esac
        g++ -O3 -fPIC -std=c++14 -DBOOST_BIND_GLOBAL_PLACEHOLDERS \
            -I"$srcdir" -c "$src" -o "$objdir/$base.o"
    done
    for src in tridiag.c tridiagv.c; do
        gcc -O3 -fPIC -std=c11 -I"$srcdir" -c "$srcdir/$src" -o "$objdir/$src.o"
    done
    g++ -O3 -fPIC -std=c++14 -DBOOST_BIND_GLOBAL_PLACEHOLDERS $define \
        -I"$srcdir" -c "$driver/abi.cpp" -o "$objdir/abi.cpp.o"
    g++ -O3 -shared -o "$libdir/libsoftsaddle_$name.so" "$objdir"/*.o -lgsl -lgslcblas
}

compile_lib lanczos "$root/upstream/libraries/lanczos_lib/to_compile" ""
compile_lib davidson "$root/upstream/libraries/davidsson_lib/to_compile" "-DSOFTSADDLE_DAVIDSON"
echo "libraries in $libdir"
ls -l "$libdir"/*.so
