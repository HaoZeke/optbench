#!/bin/bash
tar -xzf lj38-clusters.tgz

maxstep=0.1
python minimization_fire.py --maxstep=$maxstep

python minimization_getdata.py "results_data.ncalls" "Fire" pele

rm -r lj38-clusters

echo "code_file pele-$(python get_pele_version.py).tgz" >> benchmark.dat
