import numpy as np
import sys, os
import shutil
import datetime

date = datetime.date.today()
datef = date.strftime("%d %b %Y")

datafile = "results.txt"
algorithm = "hybrid eigenvector following"
code = "pele"
contributor = "Jacob Stevenson"

data = np.genfromtxt(datafile)
ncalls = []
for i in range(data.shape[0]):
    if data[i,-1] == 0:
        data[i,1] = 10000
    else:
        ncalls.append(data[i,1])
ncalls = np.array(ncalls)

with open("benchmark.dat", "w") as fout:
    fout.write( "force_calls %.5e\n" % round( np.mean(ncalls)))
    fout.write( "force_calls_median %.5e\n" % round( np.median(ncalls)))
    fout.write( "force_calls_min %.4e\n" % round( np.min(ncalls)))
    fout.write( "force_calls_max %.4e\n" % round( np.max(ncalls)))
    fout.write( "nfailed %d\n" % np.sum(1. - data[:,-1]) )
    fout.write( "algorithm %s\n" % algorithm )
    fout.write( "code %s\n" % code )
    fout.write( "contributor %s\n" % contributor )
    fout.write( "date %s\n" % datef )

