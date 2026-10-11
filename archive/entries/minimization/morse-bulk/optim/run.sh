#!/bin/bash -e

r0=2.8970
scale=`bc -l <<< "1. / $r0"`

if [ -e simulation ]; then
    rm -rf simulation
fi
mkdir simulation && cd simulation
tar xzf ../morse-bulk.tgz

for i in {0..99}
do
    file=$(printf "pos_%04d.con" $i)
    mkdir run-$i && cd run-$i
    echo run-$i
    cp ../../odata.preamble odata
    python ../../con2coords.py ../morse-bulk/$file coords --scale=$scale
    echo "points" >> odata
    awk '{print "M", $1, $2, $3}' coords >> odata
    ../../run_optim.sh $file 
    cd ..
done

python ../getdata.py

cat ../admin.dat >> benchmark.dat


mv benchmark.dat ..
cd ..


echo "code_file wales-$(OPTIM 2> /dev/null | awk '(/version/){print $3}' | sed 's/,//').tar.bz2" >> benchmark.dat
