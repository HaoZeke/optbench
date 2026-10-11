#!/usr/bin/env python3

import os
import sys
import datetime
import numpy as np
from multiprocessing import Pool

from ase.io import read

from sella import Sella

# CHANGE THIS: Number of cores to use for tests
# There are 49 tests, so set nprocs <= 49
nprocs = int(input())

# Initial guess for lowest curvature mode.
# Options are "optbench.org" and "gradient"
v0_guess = "optbench.org"

# Maximum number of gradient evaluations before giving up
maxiter = 1000

# optbench.org stipulated convergence criterion
gradtol = 1e-3

# Cutoff radius (Angstrom), value is suggested by optbench.org
cutoff = 9.5

# If you want to run only a single test, specify the test number here.
# Valid values are 0 to 48 inclusive, or None to peform all tests.
testnum = None

# Choosing which Calculator to use
# Set to False to use ASAP3 instead of LAMMPS
if True:
    # Prefer using LAMMPSlib
    #
    # This requires installing the LAMMPS Python library for the same version
    # of Python used to run Sella.
    #
    # Ask your systems administrator for help with installing the LAMMPS
    # Python library for Python 3.5+.
    from ase.calculators.lammpslib import LAMMPSlib
    Calc = LAMMPSlib
    kwargs = dict(lmpcmds=['pair_style morse/smooth/linear 9.5',
                           'pair_coeff * * 0.7102 1.6047 2.8970'],
                  log_file='test.log',
                  keep_alive=True)
else:
    # If you can't install the LAMMPS Python library, you can try using
    # the ASAP3 Morse calculator, though this seems to work more poorly.
    from asap3 import Morse
    Calc = Morse
    kwargs = dict(elements=[78],
                  epsilon=0.7102,
                  alpha=1.6047,
                  rmin=2.8970,
                  rCut=9.5)

# Sella's hyperparameters

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

# This test provides a reference minimum structure that is to be used to
# construct an initial guess for the lowest curvature mode.
atoms_ref = read('pt-island-con/reactant.con')
xref = atoms_ref.get_positions().ravel()


def run(prefix, silent=True):
#    if silent:
#        sys.stdout = open(os.devnull, 'w')

    atoms = read('pt-island-con/initial_{}.con'
                 ''.format(prefix))
    ndof = 3 * len(atoms)
    atoms.pbc = [True, True, False]

    # Check the beginning of the script to see how the Calculator is set up
    atoms.calc = Calc(**kwargs)

    if v0_guess == "optbench.org":
        v0 = atoms.get_positions().ravel() - xref
        v0 /= np.linalg.norm(v0)
    elif v0_guess == "gradient":
        v0 = None
    else:
        raise ValueError("Unknown setting for v0_guess:", v0_guess)

    dyn = Sella(atoms, v0=v0, **hyperparams)

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
    prefix = str(testnum)
    E, opt_calls, lam, gnorm, failed = run(prefix, False)
    print('{:>2}: {:20.16f} {:5d} {:20.16f} {:20.16e}'
          ''.format(prefix, E, opt_calls, lam, gnorm))
# Otherwise, do everything
else:
    results = []
    nfailed = 0
    p = Pool(nprocs)
    output = p.map(run, [str(i) for i in range(49)])
    bench = open('benchmark.dat','w')
    with open('results.txt', 'w') as f:
        for i, (E, opt_calls, lam, gnorm, failed) in enumerate(output):
            prefix = str(i)
            nfailed += failed
            results.append(opt_calls)
            prefix = str(i).zfill(4)
            f.write('{:>2}: {:20.16f} {:5d} {:20.16f} {:20.16e}\n'
                    ''.format(prefix, E, opt_calls, lam, gnorm))
    bench.write('force_calls ' + '{:.0f}'.format(np.average(results)) + '\n')
    bench.write('force_calls_min ' + str(np.min(results)) + '\n')
    bench.write('force_calls_median ' + str(np.median(results)) + '\n')
    bench.write('force_calls_max ' + str(np.max(results)) + '\n')
    bench.write('nfailed ' + str(nfailed) + '\n')
    bench.write('code Sella\ncode_version 0.1.1\ndate ' + str(datetime.datetime.today()).split()[0] + '\ncontributor Eric D. Hermes, Khachik Sargsyan, Habib Najm, Judit Zádor')
    #print('Code  |  mean  |   min  |   max ')
    #print('------+--------+--------+-------')
    #print('Sella | {:6.3f} | {:6} | {:6}'
    #      ''.format(np.average(results), np.min(results), np.max(results)))
    #print('Optim |     71 |     43 |    143')
    #print('Pele  |     88 |     52 |    198')
