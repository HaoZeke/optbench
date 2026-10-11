#!/usr/bin/env python
"""
Connect a band from dimer center to the reactant. Images on the band feels perpendicular gradient force and spring force.
Dimer force is not affected by the band. The role of the band is just to keep track of the dimer to make sure it is always 
within the basin: when the real force on dimer is in the same direction of the band tangent, it is out of the basin; move 
dimer back to the previous image along the band. 
"""

import time
import copy
import numpy as np
import sys
import os

#import ssdimer_movie
from tsase.dimer import ssdimer
from math import sqrt, atan, cos, sin, tan, pi, copysign
from ase  import atoms, units, io
from tsase.neb.util import vunit, vmag, vrand, sPBC, vproj, vdot
from tsase.io import read_con
from numpy import arctan
#from tsase.dimer import ssdimer
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS
from ase.optimize.mdmin import MDMin

def cartesianPBC(r, cell):
    icell = np.linalg.inv(cell)
    vdir  = np.dot(r, icell)
    vdir  = (vdir % 1.0 + 1.5) % 1.0 - 0.5
    newr  = np.dot(vdir, cell)
    return newr

class nebDimer_atoms:

    def __init__(self, Rmin= None, R0 = None, mode = None, maxStep = 0.2, dT = 0.1, dR = 0.005, 
                 phi_tol = 5, rotationMax = 4, ss = True, express=np.zeros((3,3)), 
                 estimateF1 = True, originalRotation = False, dTheta = 3.0, weight = 1,
                 numImages = 5, k = 5.0):
        """
        Parameters:
        Rmin - the minimum that saddles should connect to 
        R0 - starting point
        mode - initial mode (will be randomized if one is not provided)
        maxStep - longest distance dimer can move in a single iteration
        dT - quickmin timestep
        dR - finite difference step size for translation
        phi_tol - rotation converging tolerence, degree
        rotationMax - max rotations per translational step
        ss - boolean, solid-state dimer or regular dimer. Default: ssdimer
                 
        """
        self.steps   = 0
        self.normFtrans   = 1000
        self.maxStep = maxStep
        self.dT      = dT
        self.forceCalls  = 0
        self.objectCalls = 0
        self.nebCalls    = 0
        self.opt_forceCalls = 0
        self.numImages = numImages
        self.k         = k * numImages
        self.ss        = ss
        self.express   = express
        self.weight    = weight
        self.checksteps = 100
        self.quenchmax  = 3
        self.outbasin = 0
        self.outnoaction = 0
        self.checkneb = 0
        self.climb    = False

        p1    = Rmin
        p2    = R0
        self.dimer    = ssdimer.SSDimer_atoms(R0, mode = mode, maxStep = maxStep, dR = dR, phi_tol = phi_tol, 
                                       rotationMax = rotationMax, ss = ss, express = express, alpha = 1.6, alpha2 = 1.6,
                                       estimateF1 = estimateF1, originalRotation = originalRotation, dTheta = dTheta, 
                                       weight = weight
                                      )
   
        if express[0][1]**2+express[0][2]**2+express[1][2]**2 > 1e-3:
           express[0][1] = 0
           express[0][2] = 0
           express[1][2] = 0
           print "warning: xy, xz, yz components of the external pressure will be set to zero"

        #check the orientation of the cell, make sure a is along x, b is on xoy plane
        for p in [p1,p2]:
            cr = p.get_cell()
            if cr[0][1]**2+cr[0][2]**2+cr[1][2]**2 > 1e-3: 
                print "check the orientation of the cell, make sure a is along x, b is on xoy plane"
                sys.exit()
                
        #set the path by linear interpolation between end points
        self.path = [p1.copy() for i in range(self.numImages / 2)]
        self.path+= [p2]
        self.path+= [p1.copy() for i in range((self.numImages - 1) / 2)]
        calc  = p1.get_calculator()

        for i in range(self.numImages):
            # making a directory for each image, which is nessary for vasp to read last step's WAVECAR  
            # also, it is good to prevent overwriting files for parallelizaiton over images
            fdname = '0'+str(i)
            if not os.path.exists(fdname): os.mkdir(fdname)
            self.path[i].set_calculator(calc)

        #calculat the Jacobian to make cell move has the same unit and weight as atom move
        vol     = self.path[0].get_volume()
        self.natom = len(self.path[0]) 
        avglen   = (vol/self.natom)**(1.0/3.0)
        self.jacobian = avglen * self.natom**0.5 * self.weight
        self.V = np.zeros((self.numImages * (self.natom + 3), 3))
        self.Ftrans = None

        Rmin.set_calculator(R0.get_calculator())
        #dynmin = FIRE(Rmin, dt=0.2, maxmove=0.2, dtmax=0.2)
        dynmin = LBFGS(Rmin, maxstep=0.2, memory=100)
        dynmin.run(fmax = 0.001, steps = 5000)
        self.Rmin     = Rmin
        self.checkR1  = R0.copy()
        self.checkR1.set_calculator(calc)


    # Pipe all the stuff from Atoms that is not overwritten.
    # Pipe all requests for get_original_* to self.atoms0.
    def __getattr__(self, attr):
        """Return any value of the Atoms object"""
        return getattr(self.dimer, attr)
        print "*************************************"
        print attr

    def __len__(self):
        return (self.natom+3) * (self.numImages-1)

    def get_positions(self):
        Rc   = np.zeros(((self.numImages - 1) * (self.natom + 3), 3))
        return Rc

    def set_positions(self,dr):
        # because get_positions resturns all zeros, dr is got from the optimizer here
        n1 = 0
        for p in self.path:
            n2 = n1 + self.natom + 3
            dri = dr[n1:n2]
            n1 = n2

            rcell  = p.get_cell()
            rcell += np.dot(rcell, dri[-3:]) / self.jacobian
            p.set_cell(rcell, scale_atoms=True)
            ratom  = p.get_positions() + dri[:-3]
            p.set_positions(ratom)
    
    def update_general_forces(self, Ri, i):
        #update the generalized forces (f, st)
        #add some new properties

        fdname = '0'+str(i)
        if not os.path.exists(fdname): os.mkdir(fdname)
        os.chdir(fdname)
        io.write('POS_check',Ri,format='vasp')
        u    = Ri.get_potential_energy()
        f    = Ri.get_forces()
        os.chdir('../')


        vol  = Ri.get_volume()*(-1)
        st   = np.zeros((3,3))
        #following the order of get_stress in vasp.py
        #(the order of stress in ase are the same for all calculators)
        if self.ss:
            stt  = Ri.get_stress()
            st[0][0] = stt[0] * vol  
            st[1][1] = stt[1] * vol
            st[2][2] = stt[2] * vol
            st[2][1] = stt[3] * vol
            st[2][0] = stt[4] * vol
            st[1][0] = stt[5] * vol
            st  -= self.express * (-1)*vol
        #print "original stress (no projecton applied):"
        #print st
        Fc   = np.vstack((f, st/self.jacobian))
        return u, Fc
  
    def get_tangent(self, p1, p2):
        p1.cellt = p1.get_cell() * self.jacobian 
        p2.cellt = p2.get_cell() * self.jacobian 
        p1.icell = np.linalg.inv(p1.get_cell())
        p2.icell = np.linalg.inv(p2.get_cell())
        p1.vdir  = p1.get_scaled_positions()
        p2.vdir  = p2.get_scaled_positions()
        #p1.vdir  = np.dot(p1.get_positions(), p1.icell)
        #p2.vdir  = np.dot(p2.get_positions(), p2.icell)
        dr_dir = sPBC(p1.vdir - p2.vdir)
        avgbox = 0.5 * (p1.get_cell() + p2.get_cell())
        sn     = np.dot(dr_dir, avgbox)
        dh     = p1.cellt - p2.cellt
        snb    = np.dot(p1.icell, dh) * 0.5 + np.dot(p2.icell, dh) * 0.5
        ni     = np.vstack((sn, snb))
        return ni

    def get_forces(self):
        self.objectCalls += 1
        n = self.numImages 
        forces = np.zeros((n, (self.natom + 3), 3))
        energies = np.zeros(n)
        midn = self.numImages / 2 

        #print "dimer center:",self.dimer.R0.get_positions()
        if self.outbasin <= self.quenchmax:
            if self.objectCalls % self.checksteps == 1 and self.objectCalls > 1:
                #update self.outbasin
                self.checkbasin(self.Rmin)
            dimerForces   = self.dimer.get_forces()
            forces[midn]  = dimerForces
            #####need to pass to the optimizer
            self.normFtrans = vmag(forces[midn])
            self.forceCalls = self.dimer.forceCalls + self.opt_forceCalls
            print "=================still in ======================"
            print "forces mag:",self.normFtrans
            return forces.reshape((-1, 3))
        if self.outbasin > self.quenchmax:
            self.nebCalls += 1
            for i in range(n):
                energies[i], forces[i]   = self.update_general_forces(self.path[i], i)
                self.forceCalls += 1
            forces[0]  *= 0.0
            forces[-1] *= 0.0
            self.forceCalls -= 2
           
            #imax = midn
            imax = np.argsort(energies[:-1])[-1]
            self.emax  = energies[imax]
            print "energies:",energies
            #####need to pass to the optimizer
            self.normFtrans = vmag(forces[imax])

            tangent1 = self.get_tangent(self.path[1], self.path[0])
            for i in range(1, self.numImages-1):
                tangent2 = self.get_tangent(self.path[i + 1], self.path[i])
                
                lefthigh  = energies[i] < energies[i-1] 
                righthigh = energies[i] < energies[i+1]
                if lefthigh != righthigh:
                    if righthigh: tangent = copy.copy(tangent2)
                    else:         tangent = copy.copy(tangent1)
                else:
                    tangent = tangent1 + tangent2

                tangent = vunit(tangent)
                f  = forces[i]
                fpara = np.vdot(f, tangent) * tangent
                fspng = (tangent1 - tangent2) * self.k
                self.path[i].tangent   = tangent
                self.path[i].distance  = vmag(tangent1)
                self.path[i].fparaproj = np.vdot(f, tangent)
                if self.climb:
                    fspng = (vmag(tangent1) - vmag(tangent2)) * self.k * tangent
                #fspng = (vmag(tangent1) - vmag(tangent2)) * self.k * tangent
                
                if i == imax and self.climb:
                    print "================= climbing ======================"
                    print i
                    f -= 2.0 * fpara
                else:
                    f -= fspng + fpara
                    
                tangent1 = tangent2
            
            #set dimer center and the lowest mode for further reference
            self.dimer.R0 = self.path[imax]
            self.dimer.N  = self.path[imax].tangent
            print "forces norm:", vmag(forces)
            if vmag(forces) < 0.1:  
                self.climb = True
            #check if ther is intermedia minimum along the band
            if vmag(forces) < 0.5 and self.checkneb < 5 and self.nebCalls % 200 == 0:  
                for i in range(1, self.numImages-1):
                    sign = copysign(1, self.path[i].fparaproj) 
                    sign = int(sign)
                    if (i == 1 and sign < 0) or (i == self.numImages-2 and sign > 0):
                        continue
                    k = i + (sign + 1) / 2
                    dRi = self.path[k].distance
                    F1  = self.path[k - 1].fparaproj * dRi
                    F2  = self.path[k].fparaproj * dRi
                    U1  = energies[k - 1]
                    U2  = energies[k]
                    Fs  = F1 + F2
                    Ud  = U2 - U1
                    a   = U1
                    b   = -F1
                    c   = 3 * Ud + F1 + Fs
                    d   = -2 * Ud - Fs
                    #spline formula: E = d * x**3 + c * x**2 + b * x + a;  (0<=x<=1)
                    #check the curvature of x=0 and x=1
                    if c < 0.5 and 6*d + 2*c < 0.5:
                        minexist = False
                    else:
                        minexist = True

                    if minexist:
                        px0 = self.path[i].copy()
                        px0.set_calculator(self.dimer.R0.get_calculator())
                        #dyn0 = FIRE(px0, dt=0.2, maxmove=0.2, dtmax=0.2)
                        dyn0 = LBFGS(px0, maxstep=0.2, memory=100)
                        dyn0.run(fmax = 0.001, steps = 500)
                        self.forceCalls += dyn0.nsteps
                        cell = px0.get_cell()
                        dRA  = vmag(cartesianPBC(px0.get_positions() - self.Rmin.get_positions(), cell)) 
                        if dRA <= 0.3:
                            continue
                        else:
                            print "found minimum along the path"
                            self.climb     = False
                            self.nebCalls  = 0
                            self.checkneb += 1
                            rtmp0 = self.Rmin.get_positions()
                            ##### to prevent conincide
                            rtmp1 = 0.1 * self.path[i].get_positions() + 0.9 * self.path[i-1].get_positions()
                            rtmp2 = px0.get_positions()
                            drtmp  = cartesianPBC(rtmp1-rtmp0, cell)
                            rtmp01 = rtmp0 + 0.5*drtmp
                            drtmp  = cartesianPBC(rtmp2-rtmp1, cell)
                            rtmp12 = rtmp1 + 0.5*drtmp
                            self.path[0].set_positions(rtmp0) 
                            self.path[1].set_positions(rtmp01) 
                            self.path[2].set_positions(rtmp1) 
                            self.path[3].set_positions(rtmp12) 
                            self.path[4].set_positions(rtmp2) 
                            break
                    
            return forces.reshape((-1, 3))

    def checkbasin(self, pmin):

        #####needs to extend to mushy-box
        opt_ediffg = 0.001
        RDIFF = 0.3
        px0 = self.dimer.R0.copy()
        px0.set_calculator(self.dimer.R0.get_calculator())
        #dyn0 = FIRE(px0, dt=0.2, maxmove=0.2, dtmax=0.2)
        dyn0 = LBFGS(px0, maxstep = 0.2, memory=100)
        dyn0.run(fmax = opt_ediffg, steps = 10)
        cell = px0.get_cell()
        tangent = cartesianPBC(self.dimer.R0.get_positions() - px0.get_positions(), cell)
        tangent = self.get_tangent(self.dimer.R0, px0)
        checkR2 = px0.get_positions()

        dyn0.run(fmax = opt_ediffg, steps = 500)
        self.opt_forceCalls += dyn0.nsteps
        dRA  = vmag(cartesianPBC(px0.get_positions() - pmin.get_positions(), cell)) 
        F0   = self.dimer.F0
        N    = self.dimer.get_mode()
        Fpara= np.vdot(F0, N) * N
        #outdirection: True-dimer is moving outwards; False-dimer is coming back by itself.
        try: outdirection = vmag(F0) > 0.05 and np.vdot(-Fpara, self.checkR1.tangent) > 0.05 
        except: outdirection = False
        if dRA <= RDIFF:
            self.checkR1.set_positions(self.dimer.R0.get_positions())
            self.checkR1.tangent = tangent
            self.checkR1.lowestmode = copy.copy(self.dimer.N)
            self.outnoaction = 0
        elif self.outnoaction > 2 or self.outbasin < 1 or outdirection :
            print "********==========================================************"
            print "out of basin found in checkbasin:", self.outbasin
            self.outnoaction = 0
            self.outbasin+= 1
            #only get a check point when self.outbasin ==1;
            #after that, quench whenever dimer goes out until reach the max times 
            if self.outbasin == 1: 
                self.checksteps = 10
                self.dimer.R0.set_positions(self.checkR1.get_positions())
                self.dimer.R0.set_cell(self.checkR1.get_cell())
                return
            elif self.outbasin <= self.quenchmax: 
                if self.outbasin == self.quenchmax: self.checksteps = 2
                pxt = self.checkR1.copy()
                pxt.set_calculator(self.checkR1.get_calculator())
                dynt = MDMin(pxt, dt = 0.1)
                dynt.run(fmax = 0.1, steps = 1000)
                optsteps = max(dynt.nsteps / 4, 5)
                dyn1 = MDMin(self.checkR1, dt = 0.1)
                dyn1.run(fmax = opt_ediffg, steps = optsteps)
                self.opt_forceCalls += dyn1.nsteps + dynt.nsteps
                self.dimer.R0.set_positions(self.checkR1.get_positions())
                self.dimer.R0.set_cell(self.checkR1.get_cell())
                try: self.dimer.R0.N = copy.copy(self.checkR1.lowestmode)
                except: pass
                return
            #set up neb run after quenching for the max times
            rtmp0 = pmin.get_positions()
            rtmp1 = self.checkR1.get_positions()
            if self.outnoaction > 1: rtmp2 = self.neighbor.get_positions() #set the neighbor min as final state
            else:                    rtmp2 = px0.get_positions()
            drtmp  = cartesianPBC(rtmp1-rtmp0, cell)
            rtmp01 = rtmp0 + 0.5*drtmp
            drtmp  = cartesianPBC(rtmp2-rtmp1, cell)
            rtmp12 = rtmp1 + 0.5*drtmp
            self.path[0].set_positions(rtmp0) 
            self.path[1].set_positions(rtmp01) 
            self.path[2].set_positions(rtmp1) 
            self.path[3].set_positions(rtmp12) 
            self.path[4].set_positions(rtmp2) 
            #self.path[midn+1].set_positions(checkR2) 
            #self.path[midn-1].set_positions(self.checkR1.get_positions()) 
        else:
            print "********==========================================************"
            print "out of basin without any action taken:", self.outbasin
            self.outnoaction += 1
            if self.outnoaction < 2: self.neighbor = px0.copy()
            io.write('POS_check', px0, format='vasp')



#######################################################################################################
# The following part can be replaced by FIRE or MDMin optimizer in ase, see the ssdimer.py in examples
    def step(self):
        self.steps += 1
        print "************************"
        self.Ftrans = self.get_forces() 
        Ftrans = self.Ftrans
        dV = Ftrans * self.dT
        if np.vdot(self.V, Ftrans) > 0:
            self.V = dV * (1.0 + np.vdot(dV, self.V) / np.vdot(dV, dV))
        else:
            self.V = dV


        step = self.V * self.dT
        if vmag(step) > self.maxStep:
            step = self.maxStep * vunit(step)
 
        self.set_positions(step)
        self.E = self.get_potential_energy()
        
    
    def getMaxAtomForce(self):
        #Fmax = self.normFtrans
        #if Fmax is None:
        #    return 1000
        #maxForce = vmag(Fmax)
        if self.Ftrans is None:
            return 1000
        maxForce = -1
        #####only use the dimer force to determine convergence
        #for i in range((self.natom+3)*2, (self.natom+3)*3):
        for i in range(len(self.Ftrans)):
            maxForce = max(maxForce, vmag(self.Ftrans[i]))
        return maxForce
            
    def search(self, minForce = 0.01, quiet = False, maxForceCalls = 30000, movie = None, interval = 50):
        self.converged = False
        if movie:
            io.write(movie, self.dimer.R0, format='vasp')
        # While the max atom force is greater than some criteria...
        while self.getMaxAtomForce() > minForce and self.forceCalls < maxForceCalls:
            # Take a Dimer step.
            self.step()

            if movie and self.steps % interval == 0:
                traj = copy.deepcopy(self.dimer.R0)
                io.write('movie.tmp', traj, format='vasp')
                os.system('cat movie.tmp >> '+movie)

            if not quiet:
                ii = self.steps
                ff = self.getMaxAtomForce()
                cc = self.dimer.curvature
                ee = self.E
                nf = self.forceCalls 
                if self.steps % 100 == 0 or self.steps == 1:
                    print "Iteration       Force       Curvature        Energy     ForceCalls"
                    print "-------------------------------------------------------------------------------"
                    print "%3i %13.6f %13.6f %13.6f %3i" % (ii,float(ff),float(cc),float(ee),nf)
                else:
                    print "%3i %13.6f %13.6f %13.6f %3i" % (ii,float(ff),float(cc),float(ee),nf)


        if self.getMaxAtomForce() <= minForce:
            self.converged = True
                                   
