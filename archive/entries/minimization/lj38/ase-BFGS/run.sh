#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -N BFGS_1
#$ -m es
#$ -V
#$ -o ll_out
#$ -S /bin/bash
#$ -pe mpi 1

if [ -e simulation ]; then
    rm -rf simulation
fi
mkdir simulation
cp ../lj38-clusters.tgz simulation
cd simulation
tar xfz lj38-clusters.tgz
rm lj38-clusters/*.con
for i in {0..999}
do
    file=$(printf "cluster_%.4i.xyz" $i)
    mkdir run-$i
    echo run-$i
    mv lj38-clusters/$file run-$i/
    cd run-$i
    python ../../opt.py $file > stdout.dat
    cd ..
done
rm -rf lj38-clusters*
cd ..
./report.sh

