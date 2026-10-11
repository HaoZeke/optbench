#Compute the Hessian for Newton's Method and continue optimization until force cutoff is reached. 
#step size is 0.25 as default but testing it with few different values. 
#!/usr/bin/env python
import numpy as np
from numpy.linalg import eigh
from itertools import product
from ase.units import Hartree, Bohr
import scipy.linalg as la
import sys
import ase
import time
import ase.io
from sys import argv, stderr
from ase.io import read, write
np.set_printoptions(threshold=sys.maxsize)
from ase.calculators.lj import LennardJones

#Read the cluster structure from lj38 clusters.
atoms = ase.io.read(argv[1])
atoms.center(100.0)
calc = LennardJones(rc=35.0)
atoms.set_calculator(calc)

def vnorm(vector):
    """normalized vector
    """
    return vector / la.norm(vector)


def get_hessian(atoms, delta=0.0005, mass_weighted=False):
    """
    Calculate (mass weighted) hessian using central diff formula.
    :param atoms: atoms object with defined calculator
    :param delta: step size for numeric differentiation
    :type atoms: ase.Atoms
    :type delta: float
    :return: numpy square symmetric array
    """
    # convert delta to Angs
    delta *= Bohr
    # allocate matrix
    l = len(atoms)
    H = np.zeros((3 * l, 3 * l), dtype=np.float64)
    r = 0
    # gradients matrix
    for i, j in product(range(l), range(3)):
        g = np.zeros((l, 3))
        for k in (-1, 1):
            atoms1 = atoms.copy()
            atoms1[i].position[j] += k * delta
            atoms1.set_calculator(atoms.get_calculator())
            g += - k * atoms1.get_forces()

        H[r] = 0.5 * g.flatten()
        r += 1
    # check symmetry assuming gradients computed with 10^-3 Hartree/Bohr precision
    gprec = 0.001 * Hartree
    assert np.max(np.abs(H - H.T)) < gprec, np.max(np.abs(H - H.T))
    # Hessian
    H /= delta
    # symmetrize
    H = 0.5 * (H + H.T)
    # mass weight
    if mass_weighted:
        v = np.sqrt(atoms.get_masses()).repeat(3).reshape(-1, 1)
        H /= np.dot(v, v.T)
    return H
f1 = open('forces.dat', 'w')
t0 = time.time()
for i in range(100000):
    #cc = np.linalg.norm(atoms.get_forces())
    forces = atoms.get_forces()
    f1.write('forces formula from ase  %5.8f \n' %((forces**2).sum(axis=1).max()**0.5))
    f1.flush()
    #print("forces formula from ase and step number", ((forces**2).sum(axis=1).max()**0.5), i) 
    if ((forces**2).sum(axis=1).max()**0.5) < 1e-2:
        break
    Hessian = get_hessian(atoms)
    #Invert the Hessian, multiply it by the force of a given atom and displace it to find the new position and keep the loop running
    # x[i] = x[i-1] - Force/Hessian
    r = atoms.get_positions()
    f = atoms.get_forces()
    f = f.reshape(-1)
    omega, V = eigh(Hessian)
    dr = np.dot(V, np.dot(f,V)/np.fabs(omega)).reshape((-1,3))
    steplengths = (dr**2).sum(1)**0.5
    maxsteplength = np.max(steplengths)
    if maxsteplength >= 0.25:
        dr *= 0.25/maxsteplength
    atoms.set_positions(r + dr)
t1 = time.time()
f1.close()
#print("final Energy", atoms.get_potential_energy())
Hessian = get_hessian(atoms)
print('real %5.8f seconds' %(t1-t0))
write('atoms.xyz',atoms,format='xyz')
f2 = open('Hessian.txt', 'w')
f2.write(str(Hessian))
f2.flush()
f2.close()
f3 = open("results.dat", 'w')
f3.write('%i total_force_calls' %i)
f3.flush()
f3.close()
f4 = open("final_energy.dat", 'w')
f4.write('final energy is %5.8f \n' % atoms.get_potential_energy())
f4.flush()
f4.close()
