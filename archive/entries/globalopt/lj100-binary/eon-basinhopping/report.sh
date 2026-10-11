#!/bin/bash
echo runs/*/bh.dat | xargs -n 1 awk '$2 <= 2e6 {if(m>$3){m=$3}}END{print m}' | awk '{if(m>$1){m=$1}}END{print "min_energy ", m}' > benchmark.dat
echo runs/*/bh.dat | xargs -n 1 awk '$2 <= 2e6 {if(m>$3){m=$3}}END{print m}' | awk 'BEGIN{m=-99999} {if(m<$1){m=$1}}END{print "max_energy ", m}' >> benchmark.dat
echo runs/*/bh.dat | xargs -n 1 awk '$2 <= 2e6 {if(m>$3){m=$3}}END{print m}' | awk '{s+=$1}END{print "avg_energy ", s/NR}' >> benchmark.dat

echo "code Eon" >> benchmark.dat
echo "code_version r2156" >> benchmark.dat
echo "date $(date +'%d %b %Y')" >> benchmark.dat
echo "contributor Sam Chill" >> benchmark.dat
echo "code_file eon-r2156.tgz" >> benchmark.dat
