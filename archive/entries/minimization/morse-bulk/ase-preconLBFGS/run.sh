#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -N preLBFGS_1
#$ -m es
#$ -V
#$ -o ll_out
#$ -S /bin/bash
#$ -pe mpi 1
conda deactivate
if [ -e simulation ]; then
    rm -rf simulation
fi
mkdir simulation
cp ../morse-bulk.tgz simulation
cd simulation
tar xfz morse-bulk.tgz
#rm morse-bulk/*.con
for i in {0..99}
do
    file=$(printf "pos_%.4i.con" $i)
    mkdir run-$i
    echo run-$i
    mv morse-bulk/$file run-$i/
    cd run-$i
    python ../../opt.py $file > stdout.dat
    cd ..
done
rm -rf morse-bulk*
cd ..
./report.sh

