#!/bin/bash -e

if [ -e simulation ]; then
    rm -rf simulation
fi
mkdir simulation && cd simulation
tar xzf ../10000_tip4p10.tar.gz

for i in {0..9999}
do
    file=$(printf "../10000_tip4p10/%d.aa" $i)
    mkdir run-$i && cd run-$i
    echo run-$i
    cp ../../data .
    #awk '(NR>2){print $2,$3,$4}' $file | sed 's/[eE]/D/g' > coords
    cat $file > coords
    ../../run_gmin.sh $file
    cd ..
done

python ../getdata.py --datafile=results.txt

cat ../admin.dat >> benchmark.dat


mv benchmark.dat ..
cd ..

