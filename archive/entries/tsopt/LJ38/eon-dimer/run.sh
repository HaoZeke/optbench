#!/bin/sh
set -e

if [ -e runs ]; then
    rm -rf runs
fi
mkdir -p runs
cd runs
tar xfz ../coords-con.tar.gz
for i in {0..199}
do
    file=$(printf "%.4i.con" $i)
    mkdir -p run-$i
    /bin/echo -n run-$i 
    cp coords-con/$file run-$i/pos.con
    cp coords-con/$file run-$i/displacement.con
    cp ../config.ini run-$i
    cp ../in.lammps run-$i
    cd run-$i
    ../../random_mode.py $i
    eonclient > stdout.dat
    grep total_force_calls results.dat | awk '{print " ", $1}'
    cd ..
done
cd ..
./report.sh
