#!/usr/bin/env bash
set -e

if [ -e simulation ]; then
    rm -rf simulation
fi
mkdir simulation
cd simulation
tar xfz ../lj38-clusters.tgz
for i in {0..999}
do
    file=$(printf "cluster_%.4i.con" $i)
    mkdir run-$i
    echo -n "run-$i "
    mv lj38-clusters/$file run-$i/pos.con
    cp ../{config.ini,in.lammps} run-$i
    cd run-$i
    eonclient > stdout.dat
    grep force results.dat | awk '{print $1}'
    cd ..
done
rm -rf lj38-clusters
cd ..
./report.sh
