#!/bin/bash
for d in run-?
do
    cd $d
    ../done.py > benchmark.dat
    cd ..
done

echo "code Eon" > benchmark.dat
echo "code_version r2205" >> benchmark.dat
echo "date $(date +'%d %b %Y')" >> benchmark.dat
echo "contributor Sam Chill" >> benchmark.dat
echo "code_file eon-r2205.tgz" >> benchmark.dat

awk '/force/{t+=$2;n+=1}END{print "force_calls", t/n}' run-*/benchmark.dat >> benchmark.dat
awk '/force/{t+=$2;tsq+=$2*$2;n+=1}END{print "force_calls_mean_error", sqrt(tsq/n - (t/n)**2)/sqrt(n)}' run-*/benchmark.dat >> benchmark.dat
awk '/unique/{t+=$2;n+=1}END{print "unique_saddles", t/n}' run-*/benchmark.dat >> benchmark.dat
awk '/connected/{t+=$2;n+=1}END{print "fraction_connected", t/n}' run-*/benchmark.dat >> benchmark.dat
awk '/total/{t+=$2;n+=1}END{print "total_searches", t/n}' run-*/benchmark.dat >> benchmark.dat

