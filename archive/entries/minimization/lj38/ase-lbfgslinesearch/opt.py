#!/usr/bin/env python
import ase
import ase.io
import tsase
from sys import argv
import numpy as np
from time import time
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG

atoms = ase.io.read(argv[1])
atoms.center(100.0)
calc = tsase.calculators.lj(cutoff=25.0)
atoms.set_calculator(calc)

opt = ase.optimize.LBFGSLineSearch(atoms, maxstep=.2, alpha=1.0/0.001, memory=100)
#opt = ase.optimize.FIRE(atoms, maxmove=0.2, dt=0.01, dtmax=0.1)
#opt = ase.optimize.MDMin(atoms, dt=.0000001)
#opt = ase.optimize.sciopt.SciPyFminCG(atoms)
#opt = ase.optimize.sciopt.SciPyFminBFGS(atoms, alpha=33333.0)
t0 = time()
for i in xrange(10000):
    cc = np.linalg.norm(atoms.get_forces())
    if cc <= 1e-2:
        break
    opt.run(fmax=0, steps=1)
t1 = time()
print 'real %f seconds' % (t1-t0)

f = open('results.dat', 'w')
f.write('%i total_force_calls\n' % calc.force_calls)
f.close()
