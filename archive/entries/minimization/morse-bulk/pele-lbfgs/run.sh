#!/bin/bash
tar -xzf morse-bulk.tgz

M=100
maxstep=2.0
python minimization_morse.py -M $M --maxstep=$maxstep
python minimization_getdata.py "results_data.ncalls" "L-BFGS M=$M maxstep=$maxstep" pele

rm -r morse-bulk/

echo "code_file pele-$(python get_pele_version.py).tgz" >> benchmark.dat
