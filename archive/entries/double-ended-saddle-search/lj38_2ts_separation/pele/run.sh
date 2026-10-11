#!/bin/bash
tar -xzf coords.tgz

python dec_lj38_2ts.py
python getdata_ts.py
echo "algorithm DNEB + hybrid eigenvector following" >> benchmark.dat
echo "code pele" >> benchmark.dat
echo "contributor Jacob Stevenson" >> benchmark.dat


rm -r coords


echo "code_file pele-$(python get_pele_version.py).tgz" >> benchmark.dat
