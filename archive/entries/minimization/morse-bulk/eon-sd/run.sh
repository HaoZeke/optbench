#!/bin/sh
set -e

if [ -e simulation ]; then
    rm -rf simulation
fi
mkdir simulation
cd simulation
tar xfz ../morse-bulk.tgz
for i in {0..99}
do
    file=$(printf "pos_%.4i.con" $i)
    mkdir run-$i
    echo run-$i
    mv morse-bulk/$file run-$i/pos.con
    cp ../{config.ini,in.lammps} run-$i
    cd run-$i
    eonclient > stdout.dat
    cd ..
done
rm -rf morse-bulk
cd ..
./report.sh
