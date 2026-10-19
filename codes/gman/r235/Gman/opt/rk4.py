##################################################
# Functions to do 4th order Runge-Kutta  step
##################################################
import copy
import Gman.func as Gfunc
from Gman.math import sign
class rk4:
##################################################
# OPT: 4th order Runge-Kutta initialization
##################################################
  def init(self,maxmove=0.2,dt=0.05):
    """Runge-Kutta initializer function, called in script before path
    def init(self,maxmove=0.2,dR=0.0001):
    Optional arguments:
      dt:       time step
      maxmove:  maximum amount the point can move during optimization"""
    print('')
    print('4th order Runge-Kutta optimizer parameters:')
    print('  maxmove                  = ',maxmove)
    print('  dt                       = ',dt)
    self.maxmove=maxmove
    self.dt=dt
    self.dt2=dt/2.
    self.dt6=dt/6.
##################################################
# OPT: Runge-Kutta step for NEB
##################################################
  def step(self,path,force,method='min'):
    """4th order Runge-Kutta step designed to work with NEB, called from NEB
    def step(self,path,force):
    Manditory arguments:
      path:     the entire object to be minimized
      force:    potential function which sets p.U and p.F when force(p) is called
      method:   type of object to be minimized """
    
    try: path.start
    except: path.start=0
    if(not path.start):
      path.start=1
#      print 'start'
      self.k1=copy.deepcopy(path)
      self.k2=copy.deepcopy(path)
      self.k3=copy.deepcopy(path)
      self.k4=copy.deepcopy(path)
      if(method=='min'):path.ni=1
    try: path.Umaxi
    except: path.Umaxi=0

# k1 is the inital postion and force   
    for i in range(1,path.ni+1):
      if(method=='min'):i=0
      # for k1
      self.k1.p[i].R=path.p[i].R.copy()
      self.k1.Umaxi=path.Umaxi
#      force(self.k1.p[i])
      self.k1.p[i].F=path.p[i].F.copy()
      if(method=='min'):break
# should already be done in other routine if necessary
#     if(method=='neb' or method=='str' ):
#      self.k2.tangentset()
#      self.k2.project()

# position is moved by euler step with inital force
    for i in range(1,path.ni+1):
      if(method=='min'):i=0
      # for k2
      self.k2.p[i].R=path.p[i].R+self.dt2*self.k1.p[i].F
      self.k2.Umaxi=path.Umaxi
      force(self.k2.p[i])
      if(method=='min'):break
    if(method=='neb' or method=='str' ):
      self.k2.tangentset()
      self.k2.project()

# position is moved by euler step from force on fist euler step 
    for i in range(1,path.ni+1):
      if(method=='min'):i=0
      # for k3
      self.k3.p[i].R=path.p[i].R+self.dt2*self.k2.p[i].F
      self.k3.Umaxi=path.Umaxi
      force(self.k3.p[i])
      if(method=='min'):break
    if(method=='neb' or method=='str' ):
      self.k3.tangentset()
      self.k3.project()

# position is moved by euler step from force on second euler step 
    for i in range(1,path.ni+1):
      if(method=='min'):i=0
      # for k4
      self.k4.p[i].R=path.p[i].R+self.dt*self.k3.p[i].F
      self.k4.Umaxi=path.Umaxi
      force(self.k4.p[i])
      if(method=='min'):break
    if(method=='neb' or method=='str' ):
      self.k4.tangentset()
      self.k4.project()

    # move the point using the weighted average slope of four points
    for i in range(1,path.ni+1):
      if(method=='min'):i=0
      path.p[i].R +=self.dt6*(self.k1.p[i].F+2.*self.k2.p[i].F+2.*self.k3.p[i].F+self.k4.p[i].F)
      if(method=='min'):break

##################################################
##################################################
