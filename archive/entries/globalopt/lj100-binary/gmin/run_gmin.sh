#!/bin/bash -e


GMIN

# js850> note: the -a flag is necessary because some of the GMIN_out files might have binary characters which confuses grep
energy=$(head -2 lowest | tail -1 | awk '{print $5}')
ncalls=$(grep -a "Number of potential calls" output  | tail -1 | awk '{print $5}' || echo $tmp)

if [ "$energy" == "" ]; then energy="nan"; fi
if [ "$ncalls" == "" ]; then ncalls="-1"; fi

echo "$(basename $1) $energy $ncalls" >> ../results.txt
