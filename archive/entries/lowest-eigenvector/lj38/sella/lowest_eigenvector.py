#!/usr/bin/env python3

import os
import sys
import numpy as np

from ase.io import read
from ase.calculators.lj import LennardJones

from sella import PESWrapper
from multiprocessing import Pool


# DO NOT CHANGE THIS FILE
# To change settings/parameters, edit run_tests.py instead

# optbench.org provides an initial guess vector for this benchmark.
# They provide a *single* vector for *all* tests.
# We read it in here.
v0_optbench = []
with open('coords-con/initial_mode',
          'r') as f:
    for line in f:
        v0_optbench.append(float(line.strip()))
v0_optbench = np.array(v0_optbench) / np.linalg.norm(v0_optbench)


# This runs a single test and return the number of gradient evaluations
# and the leftmost eigenvalue of the approximate Hessian at convergence
def run(idx,          # Index number of structure to test
        v0,           # Initial guess for leftmost eigenvector
        eta,          # Displacement step size
        vreftol,      # optbench.org convergence tolerance
        shift,        # Use shifting instead of projection
        silent=True):
    if silent:
        sys.stdout = open(os.devnull, 'w')

    prefix = str(idx).zfill(4)

    atoms = read('coords-con/{}.con'
                 ''.format(prefix))
    atoms.pbc = False

    # Parse the true leftmost eigenvector provided by optbench.org
    vref = []
    with open('coords-con/{}_mode'
              ''.format(prefix), 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            vref.append(float(line.strip()))
    vref = np.array(vref) / np.linalg.norm(vref)

    calc = LennardJones(epsilon=1., sigma=1., rc=40.)
    pes = PESWrapper(atoms, calc, v0=v0)

    pes.diag(eta, 1e-16, vref=vref, vreftol=vreftol, shift=shift)

    opt_calls = pes.calls
    lam = pes.lams[0]

    return opt_calls, lam


def run_all(nprocs=1,
            eta=1e-6,
            v0_guess="gradient",
            zero="projection",
            vreftol=0.99):

    p = Pool(nprocs)
    args = []

    if v0_guess == "gradient":
        v0 = None
    elif v0_guess == "optbench":
        v0 = v0_optbench
    else:
        raise ValueError("Don't know v0_guess:", v0_guess)

    if zero == "projection":
        shift = False
    elif zero == "shifting":
        shift = True
    else:
        raise ValueError("Don't understand zero=", zero)

    for i in range(200):
        args.append((i, v0, eta, vreftol, shift, True))

    return p.starmap(run, args)
