#!/bin/bash -e

if [ -e simulation ]; then
    rm -rf simulation
fi
mkdir simulation && cd simulation
tar xzf ../lj100-binary-con.tgz

for i in {0..99}
do
    file=$(printf "../lj100-binary-con/coords.%d" $i)
    mkdir run-$i && cd run-$i
    echo run-$i
    cp ../../data .
    cp $file coords
    ../../run_gmin.sh $file
    cd ..
done

python ../getdata.py --datafile=results.txt

cat ../admin.dat >> benchmark.dat


mv benchmark.dat ..
cd ..


echo "code_file wales-$(OPTIM 2> /dev/null | awk '(/version/){print $3}' | sed 's/,//').tar.bz2" >> benchmark.dat
