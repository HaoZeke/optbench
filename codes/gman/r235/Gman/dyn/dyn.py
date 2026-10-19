##################################################
# Functions to do molecular dynamics
##################################################
from Gman.func import vunit,vmag
##################################################
# Molecular Dynamics Step
##################################################
def mdstep(p,force,dt=0.1,maxmove=10):
  """Takes a molecular dynamics step based on verlet-velocity
  def mdstep(p,force,dt=0.1,maxmove=10):
  Manditory arguments:
    p:        data point containing position p.R
    force:    potential function which sets p.U and p.F when force(p) is called
    dt:       dynamical timestep
  Optional arguments:
    maxmove:  maxium amount the point can move/ default is large"""

  try: p.F
  except: force(p)
  try: p.V
  except: p.V=p.R.copy()*0.0
  Rtrl=(p.V*dt+0.5*dt*dt*p.F/p.MassD)
# maxmove statment
  if(vmag(Rtrl)<maxmove):
    p.R+=Rtrl
  else:
    p.R+=maxmove*vunit(Rtrl)
  p.V+=(p.F/p.MassD)*dt*0.5
  force(p)
  p.V+=(p.F/p.MassD)*dt*0.5
##################################################
# Molecular dynamics run
##################################################
def md(p,force,dt=0.1,itrnum=100,ppd=100,vpd=0,thermV=0,bc=0):
  # Check to see if the force and velocity have been evaluated
  try: p.F
  except: force(p)
  try: p.V
  except: p.V=p.R.copy()*0.0
  print(p.V)
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

