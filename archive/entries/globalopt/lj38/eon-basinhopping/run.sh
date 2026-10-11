#!/bin/sh
set -e

if [ -e runs ]; then
    rm -rf runs
fi
mkdir -p runs
cd runs
tar xfz ../../100_lj38.tar.gz
for i in {0..99}
do
    file=$(printf "%i.con" $i)
    mkdir -p run-$i
    /bin/echo -n run-$i
    cp 100_lj38/$file run-$i/pos.con
    cp ../config.ini run-$i
    cp ../in.lammps run-$i
    cd run-$i
    eonclient > stdout.dat
    #serialsub eonclient
    grep total_force_calls results.dat | awk '{print " ", $1}'
    cd ..
done
cd ..
./report.sh
