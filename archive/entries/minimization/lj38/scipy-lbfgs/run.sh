#!/bin/bash
tar -xzf lj38-clusters.tgz

python minimization_lbfgs.py --lbfgs-scipy
python minimization_getdata.py "results_data.ncalls" "L-BFGS" SciPy

cat admin.dat >> benchmark.dat

rm -r lj38-clusters
