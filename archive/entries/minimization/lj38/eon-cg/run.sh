#!/bin/sh
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
    echo run-$i
    mv lj38-clusters/$file run-$i/pos.con
    cp ../{config.ini,in.lammps} run-$i
    cd run-$i
    eonclient > stdout.dat
    cd ..
done
rm -rf lj38-clusters
cd ..
./report.sh
