##################################################
import copy
import numpy
from Gman.func import vdot, vunit,vmag,DBC
from Gman.math import sign
class bfgs:
##################################################
# BFGS init
##################################################
  def init(self,maxmove=0.2,dR=0.0001,DFP='true',min='line',alpha=0.05,graphic=None):
    """BFGS initializer function, called in script before neb
    def init(self,maxmove=0.2,dR=0.0001,memory=4):
    Optional arguments:
      dR:       finite difference step size
      maxmove:  maximum amount the point can move during optimization
      DFP:      'true' uses faster DFP method to update the invHessian
      min:      uses a force based line minimizer"""

    print('BFGS optimizer parameters:')
    print('  maxmove                  = ',maxmove)
    print('  dR                       = ',dR)
    print('  DFP                      = ',DFP)
    self.maxmove=maxmove
    self.dR=dR
    self.DFP=DFP
    self.min=min
    self.alpha=alpha
    self.graphic=graphic

##################################################
# NEB: BFGS step for NEB
##################################################
  def step(self,path,force,method='min'):
    """BFGS step, only called from neb
    def step(self,neb,force):
    Mandatory arguments:
      neb:      the entire neb run
      force:    potential function which sets p.U and p.F when force(p) is called"""


    try:path.dim
    except: path.dim=0
    if(not path.dim):path.dim=numpy.shape(path.p[0].R)[1]

    try: path.ni
    except: path.ni=0
    if(not path.ni):path.ni=1
    try: path.Umaxi
    except: path.Umaxi=0
    if(not path.ni):path.Umaxi=1

    try: path.start
    except: path.start=0
    if(not path.start):
      path.start=1
      self.ptmp=copy.deepcopy(path)
      path.CI=0.
      path.NEXT=0.
      path.TWO_NEXT=0.
      path.OTHER=0.
      for i in range(1,path.ni+1):
        if(method=='min'):i=0
        path.p[i].bfgsinit=0
        if(method=='min'):break
    for i in range(1,path.ni+1):
      if(method=='min'):i=0
      try: path.p[i].bfgsinit
      except: path.p[i].bfgsinit=0
      if(not path.p[i].bfgsinit):
#        force(self.p[i])
        path.p[i].bfgsinit=1
        path.size=len(path.p[i].R)
        path.p[i].Bo=numpy.identity(path.dim*path.size,'d')
        if(self.min!='line'):
          path.p[i].Bo=path.p[i].Bo*self.alpha

      else:
        # check for reset
        a1=abs(vdot(path.p[i].F,path.p[i].Fold))
        a2=abs(vdot(path.p[i].Fold,path.p[i].Fold))
        if(a1<=0.5*a2 and a2!=0):
          path.p[i].G=numpy.reshape(-path.p[i].F,(path.dim*path.size,1))
          pos=numpy.reshape(DBC(path.p[i].R-path.p[i].Rold,path.p[i].Box),(path.dim*path.size,1))
          grad=path.p[i].G-path.p[i].Gold

          # Update the inverse Hessian: William Press' Numerical Recipies p. 427
          a=vdot(pos,grad)
          W=numpy.outer(pos,pos)  #matrix
          A=numpy.dot(path.p[i].Bo,grad)  #vector
          AA=numpy.outer(A,A)  #matrix
          b=vdot(grad,A)  #scaler
          if(self.DFP=='true'):
            path.p[i].Bo=path.p[i].Bo+W/a-AA/b
          else:
            u=pos/a-A/b  #vector
            uu=numpy.outer(u,u)  #matrix
            path.p[i].Bo=path.p[i].Bo+W/a-AA/b+b*uu
#            self.p[i].Bo+=W/a-AA/b+b*uu
        else:
          #print "reset",i
          path.p[i].Bo=path.p[i].Bo=numpy.identity(path.dim*path.size,'d')
          if(self.min!='line'):
            path.p[i].Bo=path.p[i].Bo*self.alpha

      path.p[i].Rold = path.p[i].R.copy()
      path.p[i].Fold = path.p[i].F.copy()
      path.p[i].Gold = numpy.reshape(-path.p[i].F,(path.dim*path.size,1))
      path.p[i].d = numpy.dot(-path.p[i].Bo,path.p[i].Gold)
      path.p[i].d3 = numpy.reshape(path.p[i].d,(path.size,path.dim))
      path.p[i].du = vunit(path.p[i].d3)

      if(self.min=='line'):
        # Finite difference step using a temporary point
        self.ptmp.p[i].R=path.p[i].R.copy()
        self.ptmp.Umaxi=path.Umaxi
#        self.ptmp.p[i].N=path.p[i].N.copy()
        self.ptmp.p[i].R+=(path.p[i].du*self.dR)
        force(self.ptmp.p[i])
      if(method=='min'):break

    if(self.min=='line'):
      # force projections on a small step
      if(method=='neb' or method=='str' ):
        self.ptmp.tangentset()
        self.ptmp.project()
      # Decide how much to move along the line Gu
      for i in range(1,path.ni+1):
        if(method=='min'):i=0
        Fp1=vdot(path.p[i].F,path.p[i].du)
        Fp2=vdot(self.ptmp.p[i].F,path.p[i].du)
        CR=(Fp1-Fp2)/self.dR
        if(CR<0.0 or  CR==0.0):
          print("neg curve")
          RdR=self.maxmove
        else:
          Fp=(Fp1+Fp2)*0.5
          RdR=Fp/CR
          if(abs(RdR)>self.maxmove):
            RdR=sign(RdR)*self.maxmove
#            print "max"
          else:
            RdR+=self.dR*0.5
        # move to the next space
        path.p[i].R+=(path.p[i].du*RdR)
        if(method=='min'):break
    else:
      for i in range(1,path.ni+1):
        if(method=='min'):i=0
        # use the Hessian Matrix to predict the minimum
        if(abs(vmag(path.p[i].d3))>self.maxmove):
          path.p[i].d3=path.p[i].du*self.maxmove
          print("maxmove")
        path.p[i].R+=path.p[i].d3
        if(method=='min'):break
    
    if(not self.graphic==None):
      self.graphic()
##################################################
##################################################
