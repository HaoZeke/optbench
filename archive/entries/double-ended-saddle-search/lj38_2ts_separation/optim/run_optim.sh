#!/bin/bash -e

cp odata.preamble odata
echo points >> odata
awk '(NR>2){print "AX", $2,$3,$4}' $1 | sed 's/[eE]/D/g' >> odata

OPTIM > OPTIM.log

ncalls=$(grep "energy+gradient calls" OPTIM.log  | tail -1 | awk '{print $6}' || echo $tmp)
#rms=$(grep "energy+gradient calls" OPTIM.log  | tail -1 | awk '{print $6}')
#energy=$(grep "mylbfgs> Final energy is" OPTIM.log  | tail -1 | awk '{print $5}')
#eigenval=$(grep "xmylbfgs> Eigenvalue and RMS=" OPTIM.log  | tail -1 | awk '{print $5}')
success=$(grep "Connected path found" OPTIM.log > /dev/null && echo 1 || echo 0)
nts=$(grep "Number of TS in the path" OPTIM.log  | tail -1 | awk '{print $8}' || echo $tmp)
ncycles=$(grep "Number of cycles" OPTIM.log  | tail -1 | awk '{print $5}' || echo $tmp)

if [ "$ncalls" == "" ]; then ncalls="-1"; fi
if [ "$nts" == "" ]; then rms="-1"; fi
if [ "$ncycles" == "" ]; then energy="-1"; fi

echo "$(basename $1) $ncalls $nts $ncycles $success" >> ../results.txt
