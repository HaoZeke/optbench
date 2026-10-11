#!/bin/bash
tar -xzf coords.tgz

python dec_lj38.py
python getdata.py

rm -r coords/

echo "code_file pele-$(python get_pele_version.py).tgz" >> benchmark.dat
