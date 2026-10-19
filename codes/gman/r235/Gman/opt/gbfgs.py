##################################################
import copy
import numpy
from Gman.func import vdot, vunit,vmag,DBC
from Gman.math import sign
class gbfgs:
##################################################
# BFGS initialization
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
    def step(self,neb,forcei,method=='min'):
    Mandatory arguments:
      path:     the entireobject to be minimized
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

   # put posion in big vector
    path.R=numpy.zeros((path.ni,len(path.p[0].R),path.dim),'d')
    for i in range(1,path.ni+1):
      if(method=='min'):
        i=0
        path.R[i]=path.p[i].R.copy()
        break
      else:
        path.R[i-1]=path.p[i].R.copy()
    # put force in big vector
    path.F=numpy.zeros((path.ni,len(path.p[0].R),path.dim),'d')
    for i in range(1,path.ni+1):
      if(method=='min'):
        i=0
        path.F[i]=path.p[i].F.copy()
        break
      else:
        path.F[i-1]=path.p[i].F.copy()

    try: path.start
    except: path.start=0
    if(not path.start):
      path.start=1
      self.ptmp=copy.deepcopy(path)
      # set maxmove to give average maxmove perimage
      self.maxmove=numpy.sqrt(self.maxmove*path.ni)
      path.bfgsinit=0
    try: path.bfgsinit
    except: path.bfgsinit=0
    if(not path.bfgsinit):
#    force(self.p[i])
      path.bfgsinit=1
      path.size=len(path.p[0].R)*path.ni
      path.Bo=numpy.identity(path.dim*path.size,'d')
      if(self.min!='line'):
        path.Bo=path.Bo*self.alpha

    else:
      if(not self.graphic==None):
        self.graphic()
      # check for reset
      a1=abs(vdot(path.F,path.Fold))
      a2=abs(vdot(path.Fold,path.Fold))
#      if(a1<=0.5*a2 and a2!=0):
      if(a2!=0):
        path.G=numpy.reshape(-path.F,(path.dim*path.size,1))
        pos=path.R-path.Rold
        # boundary conditions
        for i in range(path.ni):
          if(method=='min'):i=0
          pos[i]=DBC(pos[i],path.p[i].Box)
          if(method=='min'):break
        pos=numpy.reshape(pos,(path.dim*path.size,1))
        grad=path.G-path.Gold

        # Update the inverse Hessian: William Press' Numerical Recipies p. 427
        a=vdot(pos,grad)
        W=numpy.outer(pos,pos)  #matrix
        A=numpy.dot(path.Bo,grad)  #vector
        AA=numpy.outer(A,A)  #matrix
        b=vdot(grad,A)  #scaler
        if(self.DFP=='true'):
          path.Bo=path.Bo+W/a-AA/b
          #path.Bo+=W/a-AA/b
        else:
          u=pos/a-A/b  #vector
          uu=numpy.outer(u,u) #matrix
          path.Bo=path.Bo+W/a-AA/b+b*uu
          #path.Bo+=W/a-AA/b+b*uu
      else:
        #print "reset",i
        path.Bo=numpy.identity(path.dim*path.size,'d')
        if(self.min!='line'):
          #path.Bo=path.Bo*self.alpha
          path.Bo*=self.alpha

#    print path.Bo
    path.Rold = path.R.copy()
    path.Fold = path.F.copy()
    path.Gold = numpy.reshape(-path.F,(path.dim*path.size,1))
    path.d = numpy.dot(-path.Bo,path.Gold)
    path.d3 = numpy.reshape(path.d,numpy.shape(self.ptmp.R))
    path.du = vunit(path.d3)

    if(self.min=='line'):
      # finite difference step using temporary point
      self.ptmp.R=path.R.copy()
      self.ptmp.Umaxi=path.Umaxi
#      self.ptmp.p[i].N=path.p[i].N.copy()
      self.ptmp.R+=(path.du*self.dR)

      # put back in points
      for i in range(1,path.ni+1):
        if(method=='min'):
          i=0
          self.ptmp.p[i].R=self.ptmp.R[i]
          break
        else:
          self.ptmp.p[i].R=self.ptmp.R[i-1]
        force(self.ptmp.p[i])


    if(self.min=='line'):
      # force projections on a small step
      if(method=='neb' or method=='str' ):
        self.ptmp.tangentset()
        self.ptmp.project()

      # put force in big vector
      self.ptmp.F=numpy.zeros((path.ni,len(path.p[0].F),path.dim),'d')
      for i in range(1,path.ni+1):
        if(method=='min'):
          i=0
          self.ptmp.F[i]=self.ptmp.p[i].F
          break
        else:
          self.ptmp.F[i-1]=self.ptmp.p[i].F

      # decide how much to move along the line Gu
      Fp1=vdot(path.F,path.du)
      Fp2=vdot(self.ptmp.F,path.du)
      CR=(Fp1-Fp2)/self.dR
      if(CR<0.0):
        print("neg curve")
        RdR=self.maxmove
      else:
        Fp=(Fp1+Fp2)*0.5
        RdR=Fp/CR
        if(abs(RdR)>self.maxmove):
          RdR=sign(RdR)*self.maxmove
#          print "max"
        else:
          RdR+=self.dR*0.5
      # move to the next space
      path.R+=(path.du*RdR)
    else:
      # use the Hessian matrix to predict the minimum
      if(abs(vmag(path.d3))>self.maxmove):
        path.d3=path.du*self.maxmove
        print("maxmove")
      path.R+=path.d3

    # put back in points
    for i in range(1,path.ni+1):
      if(method=='min'):
        i=0
        path.p[i].R=path.R[i]
        break
      else:
        path.p[i].R=path.R[i-1]

    
##################################################
##################################################
