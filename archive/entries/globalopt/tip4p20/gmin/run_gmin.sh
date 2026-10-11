#!/bin/bash -e


GMIN

# js850> note: the -a flag is necessary because some of the output files might have binary characters which confuses grep
ncalls=$(grep -a "Target hit after" output  | tail -1 | awk '{print $7}' || echo $tmp)
niter=$(grep -a "Target hit after" output  | tail -1 | awk '{print $5}' || echo $tmp)
success=$(grep -a "Target hit after" output > /dev/null && echo 1 || echo 0)
rms=$(grep -a "Final Quench" output | tail -1 | awk '{print $10}')
energy=$(grep -a "Final Quench" output | tail -1 | awk '{print $5}')

if [ "$ncalls" == "" ]; then ncalls="-1"; fi
if [ "$rms" == "" ]; then rms="-1"; fi
if [ "$energy" == "" ]; then energy="-1"; fi
if [ "$niter" == "" ]; then eigenval="-1"; fi

echo "$(basename $1) $ncalls $energy $rms $niter $success" >> ../results.txt
