#!/bin/sh
set -e

if [ -e runs ]; then
    rm -rf runs
fi
mkdir -p runs
cd runs
tar xfz ../../lj100-binary-con.tgz
for i in {0..99}
do
    file=$(printf "%i.con" $i)
    mkdir -p run-$i
    /bin/echo -n run-$i
    cp lj100-binary-con/$file run-$i/pos.con
    cp ../config.ini run-$i
    cp ../in.lammps run-$i
    cd run-$i
    #eonclient > stdout.dat
    serialsub eonclient
    cd ..
done
cd ..
