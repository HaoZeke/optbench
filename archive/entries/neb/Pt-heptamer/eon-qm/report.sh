#!/bin/bash

fcalls=$(awk '$2~/force_calls_neb/{if ($1<10000){fc+=$1;n+=1}}END{printf("%.0f\n", fc/n/8)}' minima/*/results.dat)
fcalls_max=$(awk '$2~/force_calls_neb/{if ($1<10000){printf("%.0f\n", $1/8)}}' minima/*/results.dat | sort -g | tail -n 1)
fcalls_min=$(awk '$2~/force_calls_neb/{if ($1<10000){printf("%.0f\n", $1/8)}}' minima/*/results.dat | sort -gr | tail -n 1)
#nfailed=$(awk 'BEGIN{n=0} $2~/force_calls_neb/{if ($1/8>=10000){n+=1}}END{print n}' minima/*/results.dat)

echo "force_calls $fcalls" > benchmark.dat
echo "force_calls_max $fcalls_max" >> benchmark.dat
echo "force_calls_min $fcalls_min" >> benchmark.dat
#echo "nfailed $nfailed" >> benchmark.dat
echo "code Eon" >> benchmark.dat
echo "code_version r2090" >> benchmark.dat
echo "date $(date +'%d %b %Y')" >> benchmark.dat
echo "contributor Sam Chill" >> benchmark.dat
echo "code_file eon-r2090.tgz" >> benchmark.dat
