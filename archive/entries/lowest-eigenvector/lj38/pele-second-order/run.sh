#!/bin/bash
tar -xzf coords.tar.gz

python min_mode_opt.py
python minimization_getdata.py "results.dat"

cat admin.dat >> benchmark.dat

rm -r coords

echo "code_file pele-$(python get_pele_version.py).tgz" >> benchmark.dat
