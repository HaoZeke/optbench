#!/usr/bin/env python

'''
benchmark run: lj38, lowest eigenvector
1. Dimer-CG rotation
2. Dimer-BFGS rotation
3. Lanczos 
'''

from tsase.dimer import ssdimer
from tsase.io import read_con
from tsase.neb.util import vunit
from ase.calculators.lammps import LAMMPS
import numpy as np
#import pylab as pl

posfile    = 'pos.con'
modefile   = 'true_mode'
    
p1 = read_con(posfile)

pair_coeff = [' * * 1 1']
parameters = { 'pair_style':'lj/cut/opt 40.0', 'pair_coeff':pair_coeff, 'mass':['1 1']}
#calc = LAMMPS(parameters=parameters, tmp_dir= './trash')
calc = LAMMPS(parameters=parameters)
p1.set_calculator(calc)

natoms = len(p1)
mode   = np.loadtxt('direction.dat')
mode   = vunit(mode)
lowestmode  = np.loadtxt(modefile).flatten()
lowestmode  = np.reshape(lowestmode, (-1, 3))

tol = 0.99
#dimer_BFGS
for i in range(1000):
    rotmax = i + 1
    dB = ssdimer.SSDimer_atoms(p1, mode = mode, rotationMax = rotmax, phi_tol= 0.0, ss = False, dR = 0.001, rotationOpt = 'cg')
    dB.get_forces()
    dotp = np.vdot(dB.get_mode(), lowestmode)
    print rotmax, abs(dotp)
    if abs(dotp) > tol: break
BFGS_forceCalls = dB.forceCalls
print "BFGS:", BFGS_forceCalls

resultfile = open('results.dat','w')
resultfile.write(str(BFGS_forceCalls)+" total_force_calls\n")
resultfile.close()


