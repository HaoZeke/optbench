#!/bin/bash
set -e

for i in {1..10}
do
    mkdir run-$i
    cp config.ini run-$i
    cp pos.con run-$i
    cd run-$i
    #../singlerun.sh
    autosub --no-mpi ../singlerun.sh
    cd ..
done
