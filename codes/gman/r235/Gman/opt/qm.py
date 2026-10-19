##################################################
# Functions to do Quick-min step
##################################################
import copy
from Gman.func import vproj,vdot,vmag,vunit
class qm:
##################################################
# Quick-Min initialization
##################################################
  def init(self,maxmove=0.2,dt=0.15):
    """Quick-min initializer function, to be called before the neb
    def init(self,maxmove=0.2,dt=0.15):
    Optional arguments:
      dt:       dynamical timestep
      maxmove:  maximum distance that can be moved during optimization"""
    print('')
    print('Quick-min optimizer parameters:')
    print('  maxmove                  = ',maxmove)
    print('  dt                       = ',dt)
    self.maxmove=maxmove
    self.dt=dt

##################################################
# Quick-Min step
##################################################
  def step(self,path,force,method='min'):
    """quick-min step, only called from neb
    def step(self,neb,forcei,method='min'):
    Mandatory arguments:
      path:     the object to be minimized 
      force:    potential function which sets p.U and p.F when force(p) is called"""
    try: path.ni
    except: path.ni=0
    if(not path.ni): path.ni=1

    # check that the momentum is in direction of force
    for i in range(1,path.ni+1):
      if(method=='min'):i=0
      try: path.p[i].start
      except:path.p[i].start=0
      if(not path.p[i].start):
        path.p[i].start=1
        path.p[i].V=path.p[i].R.copy()*0.0

      # make sure force and velocity are in same direction (seems just as good)
      if(vdot(path.p[i].F,path.p[i].V)>0):
        path.p[i].V=vproj(path.p[i].V,path.p[i].F)
      else:
        path.p[i].V*=0.0
      #if(vdot(path.p[i].F,path.p[i].V)<0):
      #  path.p[i].V*=0.0

      # Euler step 
      path.p[i].V+=self.dt*path.p[i].F
      dR=self.dt*path.p[i].V
      # check for max step
      if(vmag(dR)>self.maxmove):
        dR=self.maxmove*vunit(dR)
#        print'maxmove'
      # move R
      path.p[i].R+=dR
      if(method=='min'):break
