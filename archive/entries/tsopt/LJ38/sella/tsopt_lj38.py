#!/usr/bin/env python3

import os
import sys

import numpy as np
import datetime

from multiprocessing import Pool

from ase.io import read
from ase.calculators.lj import LennardJones

from sella import Sella

# CHANGE THIS: Number of cores to use for tests
# There are 200 tests, so set nprocs <= 200
# Only used if all tests are to be run (if testnum is None)
nprocs = int(input())

# Maximum number of gradient evaluations before giving up
maxiter = 1000

# optbench.org stipulated convergence criterion
gradtol = 1e-3

# If you want to run only a single test, specify the test number here.
# Valid values are 0 to 199 inclusive, or None to peform all tests.
testnum = None

# Sella's hyperparameters
#
# Note: The units described below are for more realistic systems.
# This test uses a LJ potential with arbitrary units that do not correspond
# with ASE's preferred unit system (eV, Angstrom, etc.).

hyperparams = dict(
        eta=1e-6,        # Finite difference step size (Angstrom)
        gamma=0.4,       # Eigensolver convergence criterion (eV/Angstrom^2)
        rho_inc=1.035,   # Ratio threshold for increasing the trust radius
        rho_dec=5.0,     # Ratio threshold for decreasing the trust radius
        sigma_inc=1.15,  # Factor for increasing the trust radius
        sigma_dec=0.65,  # Factor for decreasing the trust radius
        delta0=1.3e-3,   # Initial trust radius (Angstrom per d.o.f.)
        )

#############################################################################
#########         Nothing should be edited below this point         #########
#########  (but feel free to read on for more explanatory comments) #########
#############################################################################


def run(prefix, silent=True):
    if silent:
        sys.stdout = open(os.devnull, 'w')

    atoms = read('coords-con/{}.con'
                 ''.format(prefix.zfill(4)))
    ndof = 3 * len(atoms)

    atoms.calc = LennardJones(epsilon=1., sigma=1., rc=40.)

    dyn = Sella(atoms, **hyperparams)

    # The convergence criterion stipulated by optbench.org is fundamentally
    # different from what ASE uses normally, so we have to bypass ASE's
    # convergence check. Fortunately, this is made easy by the irun method.
    for _ in dyn.irun(0., maxiter):
        if np.linalg.norm(atoms.get_forces()) < gradtol:
            break
    failed = 0
    if dyn.nsteps == maxiter:
        failed = 1
	

    # Number of gradient evaluations
    opt_calls = dyn.pes.calls

    # Leftmost eigenvalue of the approximate Hessian at convergence
    lam = dyn.pes.lams[0]

    # Final energy
    E = atoms.get_potential_energy()

    # Final RMS gradient
    gnorm = np.linalg.norm(atoms.get_forces()) / np.sqrt(ndof)

    return E, opt_calls, lam, gnorm, failed


# check if we're being asked for a particular test
if testnum is not None:
    prefix = str(testnum).zfill(4)
    E, opt_calls, lam, gnorm, failed = run(prefix, False)
    print('{}: {:20.16f} {:5d} {:20.16f} {:20.16e}'
          ''.format(prefix, E, opt_calls, lam, gnorm))
    print(failed)
# Otherwise, do everything
else:
    results = []
    nfailed = 0
    p = Pool(nprocs)
    bench = open('benchmark.dat','w')
    output = p.map(run, [str(i) for i in range(200)])
    with open('results.txt', 'w') as f:
        for i, (E, opt_calls, lam, gnorm, failed) in enumerate(output):
            results.append(opt_calls)
            nfailed += failed
            prefix = str(i).zfill(4)
            f.write('{}: {:20.16f} {:5d} {:20.16f} {:20.16e}\n'
                    ''.format(prefix, E, opt_calls, lam, gnorm))
    bench.write('force_calls ' + '{:.0f}'.format(np.average(results)) + '\n')
    bench.write('force_calls_min ' + str(np.min(results)) + '\n')
    bench.write('force_calls_median ' + str(np.median(results)) + '\n')
    bench.write('force_calls_max ' + str(np.max(results)) + '\n')
    bench.write('nfailed ' + str(nfailed) + '\n')
    bench.write('code Sella\ncode_version 0.1.1\ndate ' + str(datetime.datetime.today()).split()[0]+'\ncontributor Eric D. Hermes, Khachik Sargsyan, Habib Najm, Judit Zádor')
#    print('Code  |  mean  |  min  |   median  |   max   |   nfailed ')
#    print('------+--------+----------+--------+---------+-----------')
#    print('Sella | {:6} | {:6} | {:6} | {:6} | {:6}'
#          ''.format(np.average(results), np.min(results), np.median(results), np.max(results),nfailed))
#    print('Optim |    145 |   57  |           |    565  |       0')
#    print('Pele  |    192 |   59  |           |   1488  |       0')

