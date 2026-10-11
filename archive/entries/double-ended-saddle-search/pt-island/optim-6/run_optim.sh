#!/bin/bash -e

#cp odata.preamble odata
#echo points >> odata
#sed -n '3,$p' $1 | awk '{print "AX  ", $2,$3,$4}' >> odata

OPTIM > OPTIM.log

ncalls=$(grep "energy+gradient calls" OPTIM.log  | tail -1 | awk '{print $6}' || echo $tmp)
#rms=$(grep "energy+gradient calls" OPTIM.log  | tail -1 | awk '{print $6}')
#energy=$(grep "mylbfgs> Final energy is" OPTIM.log  | tail -1 | awk '{print $5}')
#eigenval=$(grep "xmylbfgs> Eigenvalue and RMS=" OPTIM.log  | tail -1 | awk '{print $5}')
success=$(grep "Converged to TS" OPTIM.log > /dev/null && echo 1 || echo 0)

if [ "$ncalls" == "" ]; then ncalls="-1"; fi
#if [ "$nts" == "" ]; then nts="-1"; fi
if [ "$ncycles" == "" ]; then ncycles="-1"; fi

echo "$(basename $1) $ncalls $success" >> ../results.txt
