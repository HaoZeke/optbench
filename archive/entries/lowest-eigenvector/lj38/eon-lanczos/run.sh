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
    mode_file=$(printf "%.4i_mode" $i)
    mkdir -p run-$i
    /bin/echo -n run-$i
    cp coords-con/$file run-$i/pos.con
    cp coords-con/$file run-$i/displacement.con
    cp ../in.lammps run-$i
    cd run-$i
    for steps in {1..1000}
    do
        sed "s/CHANGEME/$steps/" ../../config.ini > config.ini
        ../../random_mode.py $i
        eonclient > stdout.dat
        ../../compare_modes.py mode.dat ../coords-con/$mode_file && break
    done
    grep total_force_calls results.dat | awk '{print " ", $1}'
    cd ..
done
cd ..
./report.sh
