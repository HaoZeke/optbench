#!/bin/bash -e

if [ -e benchmark.dat ]; then
	rm benchmark.dat
fi

echo 'Enter number of cores for running 200 tests (<=200):'

read numcore

tar -xzvf pt-island-con.tgz

echo $numcore | python tsopt_pt-island.py
