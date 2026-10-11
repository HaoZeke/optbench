#!/bin/bash -e

if [ -e simulation ]; then
    rm -rf simulation
fi
mkdir simulation && cd simulation
tar xzf ../100_lj38.tar.gz

for i in {0..99}
do
    file=$(printf "../100_lj38/%d.xyz" $i)
    mkdir run-$i && cd run-$i
    echo run-$i
    cp ../../data .
    awk '(NR>2){print $2,$3,$4}' $file | sed 's/[eE]/D/g' > coords
    ../../run_gmin.sh $file
    cd ..
done

python ../getdata.py --datafile=results.txt

cat ../admin.dat >> benchmark.dat


mv benchmark.dat ..
cd ..


echo "code_file wales-$(OPTIM 2> /dev/null | awk '(/version/){print $3}' | sed 's/,//').tar.bz2" >> benchmark.dat
