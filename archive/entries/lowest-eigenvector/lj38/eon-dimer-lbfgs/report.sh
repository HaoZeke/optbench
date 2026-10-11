#!/bin/bash
fcalls=$(awk '$2~/total_force_calls/{if ($1<10000){fc+=$1;n+=1}}END{printf("%.0f\n", fc/n)}' runs/run-*/results.dat)
fcalls_max=$(awk '$2~/total_force_calls/{if ($1<10000){printf("%.0f\n", $1)}}' runs/run-*/results.dat | sort -g | tail -n 1)
fcalls_min=$(awk '$2~/total_force_calls/{if ($1<10000){printf("%.0f\n", $1)}}' runs/run-*/results.dat | sort -gr | tail -n 1)
#nfailed=$(awk 'BEGIN{n=0} $2~/termination_reason/&&$1!=0{n+=1}END{print n}' runs/run-*/results.dat)
nfailed=0
fcalls_median=$(grep -h force_calls runs/*/results.dat | sort -g | awk '{count[NR] = $1}END{if (NR%2) { print count[(NR+1)/2]; }else{ print (count[NR/2]+count[(NR/2)+1])/2;} }')

echo "force_calls $fcalls" > benchmark.dat
echo "force_calls_max $fcalls_max" >> benchmark.dat
echo "force_calls_min $fcalls_min" >> benchmark.dat
echo "nfailed $nfailed" >> benchmark.dat
echo "code Eon" >> benchmark.dat
echo "code_version r2156" >> benchmark.dat
echo "date $(date +'%d %b %Y')" >> benchmark.dat
echo "contributor Sam Chill" >> benchmark.dat
echo "code_file eon-r2156.tgz" >> benchmark.dat
