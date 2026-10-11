#!/usr/bin/env python
from ase.io import read
from ase.optimize import FIRE
from tsase.calculators import morse
import numpy as np
from time import time
from os import system
from sys import stdout
import datetime

system("tar xf morse-bulk.tgz")

fcs = []
for i in xrange(100):
    print 'run-%i ' % i,
    stdout.flush()
    
    atoms = read('morse-bulk/POSCAR_%i' % i)
    calc = morse(rc=9.5)
    atoms.set_calculator(calc)

    opt = FIRE(atoms, maxmove=.2, logfile=None)
    t0 = time()
    for i in xrange(1000):
        cc = np.linalg.norm(atoms.get_forces())
        if cc <= 1e-3:
            break
        opt.run(fmax=0, steps=1)
    t1 = time()

    fc = calc.force_calls
    print '%.1f' % (t1-t0), fc
    fcs.append(fc)

benchmark = {}
benchmark['force_calls'] = np.mean(fcs)
benchmark['force_calls_min'] = np.min(fcs)
benchmark['force_calls_max'] = np.max(fcs)
benchmark['code'] = 'ASE'
benchmark['code_version'] = '3.6.0'
benchmark['contributor'] = 'Sam Chill'
benchmark['code_file'] = 'python-ase-3.6.0.2515.tar.gz'
benchmark['date'] = datetime.date.today().strftime('%d %b %Y')

f = open('benchmark.dat', 'w')
for k,v in benchmark.iteritems():
    f.write('%s %s\n' % (k,v))
f.close()
