import numpy as np
import sys, os
import shutil
import datetime
import argparse

parser = argparse.ArgumentParser(description="create file benchmarks.dat from a results file for blj100",
                                 )
parser.add_argument("--datafile", type=str, help="results file to read from", 
                    default="results.txt")
args = parser.parse_args()

date = datetime.date.today()
datef = date.strftime("%d %b %Y")

datafile = args.datafile

data = np.genfromtxt(datafile)
energies = data[:,1]

emean = np.mean(energies)
emax = np.max(energies)
emin = np.min(energies)
estd = np.std(energies)

with open("benchmark.dat", "w") as fout:
    fout.write( "min_energy %.5e\n" % emin)
    fout.write( "max_energy %.5e\n" % emax)
    fout.write( "avg_energy %.5e\n" % emean)
    fout.write( "date %s\n" % datef )
