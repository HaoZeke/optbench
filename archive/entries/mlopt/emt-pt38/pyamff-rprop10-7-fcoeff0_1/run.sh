#!/bin/sh
python mlrf0_optb.py 50
python mlrf0_optb.py -r
python mlrf0_optb.py -s
echo "code Pyamff" >> benchmark.dat
echo "code_version 8037eb98" >> benchmark.dat
echo "date $(date +'%d %b %Y')" >> benchmark.dat
echo "contributor Henkelman Group" >> benchmark.dat
echo "code_file pyamff-8037eb98.tgz" >> benchmark.dat
