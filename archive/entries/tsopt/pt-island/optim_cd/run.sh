#!/bin/bash -e

rho=1.6047
r0=2.8970
rcut=9.5

scale=`bc -l <<< "1. / $r0"`
rho_scaled=`bc -l <<< "$rho * $r0"`
rcut_scaled=`bc -l <<< "$rcut / $r0"`

if [ -e simulation ]; then
    rm -rf simulation
fi
mkdir simulation && cd simulation
tar xzf ../pt-island-con.tgz

for i in {0..48}
do
    file=$(printf "initial_%d.con" $i)
    mkdir run-$i && cd run-$i
    echo run-$i
    cp ../../odata.preamble odata

    # read the coordinates, the frozen atoms, and the box lengths from the con file.  
    # also, scale the coordinates by r0 to convert to natural units
    python ../../con2coords.py ../pt-island-con/$file coords --scale=$scale --frozen=frozen --box-lengths=boxvec
    cp ../pt-island-con/$file fort.11
    cp ../pt-island-con/reactant.con fort.12
    ../../genmode.x
    # write the box lenghts to the odata file
    boxvec=`cat boxvec`
    echo "PARAMS $rho_scaled $boxvec $rcut_scaled" >> odata

    # add the frozen atoms the the odata file
    awk '{print "FREEZE ", $0}' frozen >> odata

    # add the coordinates to the odata file
    echo "points" >> odata
    awk '{print "M", $1, $2, $3}' coords >> odata


    # run OPTIM and parse the results
    ../../run_optim.sh $file 
    cd ..
done

python ../getdata.py

cat ../admin.dat >> benchmark.dat


mv benchmark.dat ..
cd ..


echo "code_file wales-$(OPTIM 2> /dev/null | awk '(/version/){print $3}' | sed 's/,//').tar.bz2" >> benchmark.dat
