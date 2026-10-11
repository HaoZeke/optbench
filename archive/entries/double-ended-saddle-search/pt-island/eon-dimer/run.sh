#!/bin/sh
set -e

if [ -e runs ]; then
    rm -rf runs
fi
mkdir -p runs
cd runs
tar xfz ../pt-island-con.tgz
for i in {0..58}
do
    product=$(printf "product_%i.con" $i)
    mkdir -p run-$i
    /bin/echo -n run-$i
    cp pt-island-con/reactant.con run-$i/pos.con
    ../interpolate.py pt-island-con/reactant.con pt-island-con/$product run-$i/displacement.con
    cp ../config.ini run-$i
    cd run-$i
    ../../make_mode.py
    eonclient > stdout.dat
    grep total_force_calls results.dat | awk '{print " ", $1}'
    cd ..
done
cd ..
./report.sh
