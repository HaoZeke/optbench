#!/bin/bash
fcalls=$(awk '$2~/total_force_calls/{fc+=$1;n+=1}END{printf("%.0f\n", fc/n)}' runs/run-*/results.dat)
fcalls_max=$(awk '$2~/total_force_calls/{printf("%.0f\n", $1)}' runs/run-*/results.dat | sort -g | tail -n 1)
fcalls_min=$(awk '$2~/total_force_calls/{printf("%.0f\n", $1)}' runs/run-*/results.dat | sort -gr | tail -n 1)

echo "force_calls $fcalls" > benchmark.dat
echo "force_calls_max $fcalls_max" >> benchmark.dat
echo "force_calls_min $fcalls_min" >> benchmark.dat
echo "code Eon" >> benchmark.dat
echo "code_version r2156" >> benchmark.dat
echo "date $(date +'%d %b %Y')" >> benchmark.dat
echo "contributor Sam Chill" >> benchmark.dat
echo "code_file eon-r2156.tgz" >> benchmark.dat
