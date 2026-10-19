##################################################
# Functions to do Fast Inertial Relaxation Engine step
##################################################
from Gman.func import vproj,vdot,vmag,vunit
import copy
class fire:
##################################################
# Fast Inertial Relaxation Engine initialization
##################################################
    def init(self,maxmove=0.2,dt=0.1,dtmax=1.0,Nmin=5,finc=1.1,fdec=0.5,astart=0.1,fa=0.99):
      """Fire initializer function, called in script before min
      def init(self,maxmove=0.2,dtmax=0.15,Nmin=5,finc=1.1,fdec=0.5,astart=0.1,fa=0.99):
      Optional arguments:
        dt:       dynamical timestep
        maxmove:  maximum amount the point can move during optimization"""
      print('Fire optimizer parameters:')
      print('  maxmove                  = ',maxmove)
      print('  dt                       = ',dt)
      print('  dtmax                    = ',dtmax)
      print('  Nmim                     = ',Nmin)
      print('  finc                     = ',finc)
      print('  fdec                     = ',fdec)
      print('  astart                   = ',astart)
      print('  fa                       = ',fa)
      self.maxmove=maxmove
      self.dt=dt
      self.dtmax=dtmax
      self.Nmin=Nmin
      self.finc=finc
      self.fdec=fdec
      self.astart=astart
      self.fa=fa

##################################################
# Fast Inertial Relaxation Engine step
##################################################
    def step(self,path,force,method='min'):
      """fire step, typically called from min
      def step(self,p,force):
      Mandatory arguments:
        p:        data point containing position p.R
        force:    potential function which sets p.U and p.F when force(p) is called"""

      try:path.ni
      except:path.ni=0
      if(not path.ni):path.ni=1

      try: path.start
      except: path.start=0
      if(not path.start):
        path.start=1
        for i in range(1,path.ni+1):
          if(method=='min'):i=0
          path.p[i].V=path.p[i].F.copy()*0.0
          path.p[i].a=self.astart
          path.p[i].Nsteps=0
          path.p[i].dt=self.dt
          if(method=='min'):break

      for i in range(1,path.ni+1):
        if(method=='min'):i=0
        Power = vdot(path.p[i].F,path.p[i].V)
        if(Power>0.0):
          path.p[i].V = (1.0-path.p[i].a)*path.p[i].V+path.p[i].a*vmag(path.p[i].V)*vunit(path.p[i].F)
          if(path.p[i].Nsteps>self.Nmin):
            path.p[i].dt = min(path.p[i].dt*self.finc,self.dtmax)
            path.p[i].a *= self.fa
          path.p[i].Nsteps+=1
        else:
          # reset velocity and slow down
          path.p[i].V *= 0.0
          path.p[i].a = self.astart
          path.p[i].dt *= self.fdec  
          path.p[i].Nsteps = 0

#      for i in range(p.NumAtoms):
#        if(vdot(p.V[i],p.F[i])>0):
#          p.V[i]=vproj(p.V[i],p.F[i])
#        else:
#          p.V[i]*=0.0

        # Euler step 
        path.p[i].V+=self.dt*path.p[i].F
        dR=self.dt*path.p[i].V
        # check for max step
        if(vmag(dR)>self.maxmove):
          dR=self.maxmove*vunit(dR)
#          print'maxmove'
      # move R
        path.p[i].R+=dR
        if(method=='min'):break

