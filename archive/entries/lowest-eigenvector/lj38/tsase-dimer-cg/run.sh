#!/bin/sh
set -e

if [ -e runs ]; then
    rm -rf runs
fi
mkdir -p runs
cd runs
tar xfz ../coords-con.tar.gz
for i in {0..199}
do
    file=$(printf "%.4i.con" $i)
    mode_file=$(printf "%.4i_mode" $i)
    mkdir -p run-$i
    /bin/echo -n run-$i
    cp coords-con/$file run-$i/pos.con
    cp coords-con/$mode_file run-$i/true_mode
    cp ../*.py run-$i
    cd run-$i
    ../../random_mode.py $i
    ## serial 
    #python dimer_rotate.py $i  > stdout.dat
    ## submit jobs
    wkdir=$(pwd)
    printf $wkdir
    qsub -V -b y -j y -o ll_out -S /bin/csh -N "run-$i" -wd "$wkdir" dimer_rotate.py 
    #grep total_force_calls results.dat | awk '{print " ", $1}'
    cd ..
done
cd ..
#./report.sh
