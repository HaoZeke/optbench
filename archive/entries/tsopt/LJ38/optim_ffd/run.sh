#!/bin/bash -e

if [ -e simulation ]; then
    rm -rf simulation
fi
mkdir simulation && cd simulation
tar xzf ../coords-xyz.tar.gz

for i in {0..199}
do
    file=$(printf "%.4i.xyz" $i)
    mkdir run-$i && cd run-$i
    echo run-$i
    cp ../../odata.preamble .
    ../../run_optim.sh ../coords-xyz/$file 
    cd ..
done

python ../getdata.py

cat ../admin.dat >> benchmark.dat

mv benchmark.dat ..
cd ..


echo "code_file wales-$(OPTIM 2> /dev/null | awk '(/version/){print $3}' | sed 's/,//').tar.bz2" >> benchmark.dat
