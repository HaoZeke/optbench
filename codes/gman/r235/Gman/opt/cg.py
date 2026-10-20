##################################################
# Functions to do conjugate gradient step
##################################################
import copy
import Gman.func as Gfunc
from Gman.math import sign
class cg:
##################################################
# NEB: conjugate gradient initialization 
##################################################
  def init(self,maxmove=0.2,dR=0.0001):
    """Conjugate gradient initializer function, called in script before path
    def init(self,maxmove=0.2,dR=0.0001):
    Optional arguments:
      dR:       finite difference step size
      maxmove:  maximum amount the point can move during optimization"""
    print('')
    print('Conjugate Gradient optimizer parameters:')
    print('  maxmove                  = ',maxmove)
    print('  dR                       = ',dR)
    self.maxmove=maxmove
    self.dR=dR
##################################################
# NEB: conjugate gradient step for NEB
##################################################
  def step(self,path,force,method='min'):
    """conjugate gradient step designed to work with NEB, called from NEB
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
      self.ptmp=copy.deepcopy(path)
      path.CI=0.
      path.NEXT=0.
      path.TWO_NEXT=0.
      path.OTHER=0.
      if(method=='min'):path.ni=1
      for i in range(1,path.ni+1):
        if(method=='min'):i=0
        path.p[i].cginit=0
        if(method=='min'):break
    try: path.Umaxi
    except: path.Umaxi=0

    for i in range(1,path.ni+1):
      if(method=='min'):i=0
      try:path.p[i].cginit
      except:path.p[i].cginit=0
      if(not path.p[i].cginit):
        path.p[i].cginit=1
        path.p[i].gam=0
        path.p[i].d=path.p[i].F.copy()
        path.p[i].Fold=path.p[i].F.copy()

      # determine CG parameters
      a1=abs(Gfunc.vdot(path.p[i].F,path.p[i].Fold))
      a2=abs(Gfunc.vdot(path.p[i].Fold,path.p[i].Fold))
      if(a1<=0.5*a2 and a2!=0):
        gam=Gfunc.vdot(path.p[i].F,(path.p[i].F-path.p[i].Fold))/a2
      else:
#        print 'reset'
        gam=0
      
#      gam=0

      path.p[i].d=path.p[i].F+path.p[i].d*gam
      path.p[i].du=Gfunc.vunit(path.p[i].d)
      path.p[i].Fold=path.p[i].F.copy()
      # finite difference step using a temporary point
      self.ptmp.p[i].R=path.p[i].R.copy()
      self.ptmp.Umaxi=path.Umaxi
#      self.ptmp.p[i].N=path.p[i].N.copy()
      self.ptmp.p[i].R+=(path.p[i].du*self.dR)
      force(self.ptmp.p[i])
      # break loop if only 1 image
      if(method=='min'):break

    # force projections on small step
    if(method=='neb' or method=='str' ):
      self.ptmp.tangentset() # helps a little
      self.ptmp.project()   # I think this is correct
    for i in range(1,path.ni+1):
      if(method=='min'):i=0
     # decide how much to move along the line Gu
      Fp1=Gfunc.vdot(path.p[i].F,path.p[i].du)
      Fp2=Gfunc.vdot(self.ptmp.p[i].F,path.p[i].du)
      CR=(Fp1-Fp2)/self.dR
#      print CR
      if(CR<0.0):
#        print CR, i
#        print 'neg curve'
        if(i==path.Umaxi):
          print("neg curv",i,"Climbing Image")
          path.CI+=1
        else:
          if(i+1==path.Umaxi or i-1==path.Umaxi):
            print("negcurv",i,"next to ci")
            path.NEXT+=1
          else:
            if(i+2==path.Umaxi or i-2==path.Umaxi):
              print("negcurv",i,"2 images away from max energy")
              path.TWO_NEXT+=1
            else:
              print("neg curv",i,"other")
              path.OTHER+=1
        RdR=self.maxmove 
      else:
#        Fp=(Fp1+Fp2)*0.5
        # weighted average
        alpha=Fp1/(Fp1+Fp2)
        Fp=alpha*Fp1+(1-alpha)*Fp2
        CR=(Fp1-Fp2)/self.dR
        RdR=Fp/CR
        if(abs(RdR)>self.maxmove):
#          print 'image',i,'maxmove'
          RdR=sign(RdR)*self.maxmove
        else:
          RdR+=(1-alpha)*self.dR

      # move the point 
      path.p[i].R+=(path.p[i].du*RdR)
      # break loop if only 1 image
      if(method=='min'):break

##################################################
##################################################
