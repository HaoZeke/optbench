#!/bin/bash

tar -xzf 100_lj38.tar.gz
gunzip -c lj38_gmin.con.gz > lj38_gmin.con

python globalopt.py

rm -r 100_lj38
rm lj38_gmin.con

echo "code_file pele-$(python get_pele_version.py).tgz" >> benchmark.dat
