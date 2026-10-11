#!/bin/bash -e

if [ -e simulation ]; then
    rm -rf simulation
fi
mkdir simulation && cd simulation
tar xzf ../lj38-clusters.tgz

for i in {0..999}
do
    file=$(printf "cluster_%04d.xyz" $i)
    mkdir run-$i && cd run-$i
    echo run-$i
    cp ../../odata.preamble .
    ../../run_optim.sh ../lj38-clusters/$file 
    cd ..
done

python ../getdata.py

cat ../admin.dat >> benchmark.dat


mv benchmark.dat ..
cd ..

echo "code_file wales-$(OPTIM 2> /dev/null | awk '(/version/){print $3}' | sed 's/,//').tar.bz2" >> benchmark.dat
