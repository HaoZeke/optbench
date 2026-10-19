##################################################
# Dimer Method calculation
##################################################

from Gman.data import data
from Gman.func import vmag, vproj, vdot, vunit
from math import sqrt, atan, pi
from Gman.neb import qm

class dimer:
##################################################
# Dimer 
##################################################
def step(p,force,dt=0.1):
  try: p.F
  except: force(p)
  try: p.V
  except: p.V=p.R.copy()*0.0
  p.R+=(p.V*dt+0.5*dt*dt*p.F/p.MassD)
  p.V+=(p.F/p.MassD)*dt*0.5
  force(p)
  p.V+=(p.F/p.MassD)*dt*0.5
##################################################
# Dimer run
##################################################
def dim(p,force,dt=0.1,itrnum=100,ppd=100,vpd=0,thermV=0,bc=0):
  # Check to see if the force and velocity have been evaluated
  try: p.F
  except: force(p)
  try: p.V
  except: p.V=p.R.copy()*0.0
  print p.V
  for i in range(itrnum):
    # first check for a theromostat
    if(vpd!=0):
      if(i%vpd==0):
        try: p.V=thermV()
        except: ''#print 'No thermostat set'
    # velocity verlet algorithm
    mdstep(p,force,dt)
    # apply boundary conditions
    try: Gfunc.BC(p.R,p.Box,p.iBox)
    except: ''# do nothing

