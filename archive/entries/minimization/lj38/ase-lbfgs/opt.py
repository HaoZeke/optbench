#!/usr/bin/env python
import ase
import ase.io
import tsase
from sys import argv, stderr
import numpy as np
from time import time

atoms = ase.io.read(argv[1])
atoms.center(100.0)
calc = tsase.calculators.lj(cutoff=35.0)
atoms.set_calculator(calc)

opt = ase.optimize.LBFGS(atoms, maxstep=.1, alpha=1.0/0.001, memory=100)
t0 = time()
for i in xrange(10000):
    cc = np.linalg.norm(atoms.get_forces())
    if cc <= 1e-2:
        break
    opt.run(fmax=0, steps=1)
t1 = time()
print 'real %f seconds' % (t1-t0)
stderr.write('%s\n'%calc.force_calls)

f = open('results.dat', 'w')
f.write('%i total_force_calls\n' % calc.force_calls)
f.close()
