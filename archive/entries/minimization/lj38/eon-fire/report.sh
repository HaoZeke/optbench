#!/bin/bash
if [ ! -d simulation ]; then
    tar xfj simulation.tar.bz2
fi

fcalls=$(awk '$2~/total_force_calls/{if ($1<10000){fc+=$1;n+=1}}END{printf("%.0f\n", fc/n)}' simulation/run-*/results.dat)
fcalls_max=$(awk '$2~/total_force_calls/{if ($1<10000){printf("%.0f\n", $1)}}' simulation/run-*/results.dat | sort -g | tail -n 1)
fcalls_min=$(awk '$2~/total_force_calls/{if ($1<10000){printf("%.0f\n", $1)}}' simulation/run-*/results.dat | sort -gr | tail -n 1)
nfailed=$(awk 'BEGIN{n=0} $2~/total_force_calls/{if ($1>=10000){n+=1}}END{print n}' simulation/run-*/results.dat)
walltime=$(awk '$1 ~ /real/ && $3 ~ /seconds/ {sum+=$2;n+=1}END{printf("%.4f\n", sum/n)}' simulation/run-*/stdout.dat)

echo "force_calls $fcalls" > benchmark.dat
echo "force_calls_max $fcalls_max" >> benchmark.dat
echo "force_calls_min $fcalls_min" >> benchmark.dat
fc_per_second=$(awk "BEGIN {print $fcalls / $walltime}")
echo "force_calls_per_second $fc_per_second" >> benchmark.dat
echo "nfailed $nfailed" >> benchmark.dat
echo "algorithm FIRE" >> benchmark.dat
echo "code Eon" >> benchmark.dat
echo "code_version r2025" >> benchmark.dat
echo "date $(date +'%d %b %Y')" >> benchmark.dat
echo "contributor Sam Chill" >> benchmark.dat
echo "code_file eon-r2025.tgz" >> benchmark.dat
