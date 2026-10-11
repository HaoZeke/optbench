#!/bin/bash -e

if [ -e simulation ]; then
    rm -rf simulation
fi
mkdir simulation && cd simulation
source_dir="ljj8_clusters"
tar xzf ../../${source_dir}.tgz

for i in {1..200}
do
    file=$(printf "../${source_dir}/coords.%d" $i)
    mkdir run-$i && cd run-$i
    echo run-$i
    cp ../../data .
    #awk '(NR>2){print $2,$3,$4}' $file | sed 's/[eE]/D/g' > coords
    cp $file coords
    ../../run_gmin.sh $file
    cd ..
done

python ../getdata.py --datafile=results.txt

cat ../admin.dat >> benchmark.dat


mv benchmark.dat ..
cd ..

