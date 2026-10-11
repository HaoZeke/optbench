#!/usr/bin/env python

'''
SSDimer and optimization used for akmc in eOn.
set client_path = ./ssdimer_ase.py
'''

from tsase.calculators.morse import morse
from tsase.dimer import ssdimer
from tsase.neb.util import vunit, vrand, vmag
from tsase.io import read_con, write_con
from ase.io import write, read
from ase.optimize.fire import FIRE
from ase.optimize.mdmin import MDMin
import os
import sys
import numpy as np

alpha        = 0.6
alpha2       = 2.5
dimer_ediffg = 0.0001
opt_ediffg   = 0.0001
#energy tolerence for state comparison
EDIFF        = 0.01
RDIFF        = 0.3

# set initial geometry and calculator
p1        = read_con('pos.con')
calc      = morse()
p1.set_calculator(calc)
Ereactant = p1.get_potential_energy()
print "Ereactant:", Ereactant
write_con('reactant.con', p1)

p2        = read_con('displacement.con')
p2.set_calculator(calc)
displacedis = vmag(p2.get_positions() - p1.get_positions())
mode      = np.loadtxt('direction.dat')
mode      = np.vstack(( mode, np.zeros((3,3)) ))


##############################################
# start dimer search
status = -1
d = ssdimer.SSDimer_atoms(p2, mode = mode, rotationMax = 50, maxStep = 0.20, phi_tol= 5, alpha = alpha, alpha2 = alpha2, ss = False)
d.search(minForce = dimer_ediffg, movie = "dimerSearch.movie", interval = 50 )
#dyndimer = MDMin(d)
#dyndimer.run(fmax = dimer_ediffg, steps=50)
#dyndimer = FIRE(d,dt=0.1, maxmove=0.1, dtmax=0.1)
#dyndimer.run(fmax = dimer_ediffg, steps = 5000)
dimer_forceCalls = d.forceCalls
#if vmag(d.get_forces()) > dimer_ediffg*3: 
if dimer_forceCalls > 100000:
    status = 5 # maximum steps
mode = d.get_mode()
write_con("saddle.con", d.R0)
np.savetxt('mode.dat', mode)
Esaddle = d.get_potential_energy()
print "barrier:",Esaddle - Ereactant

##############################################
#start optimization part
natom = len(p1)
vol   = p1.get_volume()
jacob = (vol/natom)**(1.0/3.0) * natom**0.5
def set_endpoint_pos(Ni, R0, Ri, dR=1.0):
    # displace from R0 along direction Ni
    dRvec = dR * Ni
    cell0 = R0.get_cell()
    cell1 = cell0 + np.dot(cell0, dRvec[-3:]) / jacob
    Ri.set_cell(cell1, scale_atoms=True)
    vdir  = R0.get_scaled_positions()
    ratom = np.dot(vdir, cell1) + dRvec[:-3]
    Ri.set_positions(ratom)

if status != 5:
    px1    = d.R0.copy()
    px0    = px1.copy()
    px2    = px1.copy()
    px0.set_calculator(calc)
    px2.set_calculator(calc)
    set_endpoint_pos( mode, px1, px0)
    set_endpoint_pos(mode*(-1), px1, px2)

    dyn0 = FIRE(px0, dt=0.2, maxmove=0.2, dtmax=0.2)
    #dyn0 = MDMin(p0box)
    dyn0.run(fmax = opt_ediffg, steps = 5000)

    dyn2 = FIRE(px2, dt=0.2, maxmove=0.2, dtmax=0.2)
    #dyn2 = MDMin(p2box)
    dyn2.run(fmax = opt_ediffg, steps = 5000)

    opt_forceCalls = dyn0.nsteps + dyn2.nsteps

    # check the state of the optimized structure
    dEA  = px0.get_potential_energy() - Ereactant 
    dEB  = px2.get_potential_energy() - Ereactant 
    dRA  = vmag(px0.get_positions() - p1.get_positions()) 
    dRB  = vmag(px2.get_positions() - p1.get_positions()) 
    #tmpA = (abs(dEA) <= EDIFF)  # can be updated to a comparison function
    #tmpB = (abs(dEB) <= EDIFF)
    tmpA = (dRA <= RDIFF)  
    tmpB = (dRB <= RDIFF)
    if tmpA and tmpB:
        status = 6  # not connectted
        print "both connectted to the initial state"
    elif tmpA:
        status = 0  # correct
        write_con("product.con",px2)
        Eproduct = px2.get_potential_energy()
        print "product energy:", Eproduct, Eproduct-Ereactant
    elif tmpB:
        status = 0
        write_con("product.con",px0)
        Eproduct = px0.get_potential_energy()
        print "product energy:", Eproduct, Eproduct-Ereactant
    else:
        status = 6  # not connectted
        print "not connectted to the initail state"
   
    print dEA, dEB
    write_con("CONT_backward.con",px0)
    write_con("CONT_forward.con",px2)
else:
    opt_forceCalls = 0

#################################################################
#output
total_forceCalls = dimer_forceCalls + opt_forceCalls
if status != 0:
    Ereactant = 0.0
    Eproduct  = 0.0
    Esaddle   = 0.0

resultfile = open('results.dat','w')
resultfile.write(str(status)+" termination_reason\n")
resultfile.write("saddle_search job_type\n")
resultfile.write(str(total_forceCalls)+" total_force_calls\n")
resultfile.write(str(dimer_forceCalls)+" force_calls_saddle\n")
resultfile.write(str(opt_forceCalls)+" force_calls_minimization\n")

resultfile.write(str(displacedis)+" displacement_saddle_distance\n")
resultfile.write(str(Ereactant)+" potential_energy_reactant\n")
resultfile.write(str(Esaddle)+" potential_energy_saddle\n")
resultfile.write(str(Eproduct)+" potential_energy_product\n")
resultfile.write(str(Esaddle-Ereactant)+" barrier_reactant_to_product\n")
resultfile.write(str(Esaddle-Eproduct)+" barrier_product_to_reactant\n")
   
#fake
resultfile.write(str(0)+" iterations\n")
resultfile.write(str(42)+" random_seed\n")
resultfile.write("eam_al potential_type\n")
resultfile.write(str(1.2e12)+" prefactor_product_to_reactant\n")
resultfile.write(str(1.2e12)+" prefactor_reactant_to_product\n")
resultfile.write(str(0)+" force_calls_prefactors")
resultfile.close()


