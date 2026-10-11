#!/bin/bash -e

if [ -e simulation ]; then
    rm -rf simulation
fi
mkdir simulation && cd simulation
tar xzf ../coords.tgz

for i in {0..49}
do
    f1=$(printf "start_%02d.xyz" $i)
    f2=$(printf "end_%02d.xyz" $i)
    mkdir run-$i && cd run-$i
    echo run-$i $f1 $f2
    cp ../../odata.preamble .
    cp odata.preamble odata
    
    awk '(NR>2){print $2, $3, $4}' ../coords/$f2  | sed 's/[eE]/D/g' > finish
    ../../run_optim.sh ../coords/$f1 
    cd ..
done

python ../getdata.py

echo "algorithm DNEB + hybrid eigenvector following" >> benchmark.dat
echo "code OPTIM" >> benchmark.dat
echo "contributor Jacob Stevenson & Cheng Shang" >> benchmark.dat


mv benchmark.dat ..
cd ..


echo "code_file wales-$(OPTIM 2> /dev/null | awk '(/version/){print $3}' | sed 's/,//').tar.bz2" >> benchmark.dat
