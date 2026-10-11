#!/bin/bash

if [ -e benchmark.dat ]; then
	rm benchmark.dat
fi

echo 'Enter number of processors for running 200 tests (<=200):'

read numcore

tar -xzvf coords-con.tar.gz

echo $numcore | python lowest-eigv.py
