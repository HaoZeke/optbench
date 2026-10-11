#!/bin/bash
tar -xzf lj38-clusters.tgz

M=4
maxstep=0.2

python minimization_lbfgs.py -M $M --maxstep=$maxstep
python minimization_getdata.py "results_data.ncalls" "L-BFGS M=$M maxstep=$maxstep" pele

cat admin.dat >> benchmark.dat

rm -r lj38-clusters

echo "code_file pele-$(python get_pele_version.py).tgz" >> benchmark.dat
