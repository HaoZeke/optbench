#!/bin/sh
set -e

if [ -e runs ]; then
    rm -rf runs
fi
mkdir -p runs
cd runs
tar xfz ../coords-con.tgz
for i in {0..49}
do
    reactant=$(printf "start_%.2i.con" $i)
    product=$(printf "end_%.2i.con" $i)
    mkdir -p run-$i
    /bin/echo -n run-$i
    cp coords-con/$reactant run-$i/reactant.con
    cp coords-con/$product run-$i/product.con
    cp ../config.ini run-$i
    cp ../in.lammps run-$i
    cd run-$i
    eonclient > stdout.dat
    grep total_force_calls results.dat | awk '{print " ", $1}'
    cd ..
done
cd ..
./report.sh
