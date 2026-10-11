#!/bin/sh
set -e

if [ -e runs ]; then
    rm -rf runs
fi
mkdir -p runs
cd runs
tar xfz ../pt-island-con.tgz
for i in {0..48}
do
    file=$(printf "initial_%i.con" $i)
    mkdir -p run-$i
    /bin/echo -n run-$i
    cp pt-island-con/reactant.con run-$i/pos.con
    cp pt-island-con/$file run-$i/displacement.con
    cp ../config.ini run-$i
    cd run-$i
    ../../random_mode.py $i
    eonclient > stdout.dat
    grep total_force_calls results.dat | awk '{print " ", $1}'
    cd ..
done
cd ..
./report.sh
