##################################################
import copy
import numpy
from Gman.func import vdot,vunit,vmag,DBC
from Gman.math import sign
class glbfgs:
##################################################
# NEB: LBFGS initilize
##################################################
  def init(self,maxmove=0.2,dR=0.0001,memory=25,min='line',alpha=0.05):
    """Limited memory bfgs initilizer function, called in script before neb
    def init(self,maxmove=0.2,dR=0.0001,memory=4,min='line'):
    Optional arguments:
      dR:       finite difference step size
      maxmove:  maximum amount the point can move during optimization
      memory:   number of preceeding steps remembered"""
    print('')
    print('Limited-memory BFGS optimizer parameters:')
    print('  maxmove                  = ',maxmove)
    print('  dR                       = ',dR)
    print('  memory                   = ',memory)
    print('  minimizer                = ',min)
    self.maxmove=maxmove
    self.dR=dR
    self.memory=memory
    self.min=min
    self.alpha=alpha

##################################################
# NEB: LBFGS step for NEB
##################################################
  def step(self,path,force,method='min'):
    """Limited-memory BFGS step, only called from neb
    def step(self,neb,force):
    Manditory arguments:
      neb:      the entire neb run
      force:    potential function which sets p.U and p.F when force(p) is called"""
    try: path.ni
    except: path.ni=0
    if(not path.ni):path.ni=1
    try: path.Umaxi
    except: path.Umaxi=0
    if(not path.ni):path.Umaxi=1

    try:path.dim
    except: path.dim=0
    if(not path.dim):path.dim=numpy.shape(path.p[0].R)[1]

    # put posion in big vector
    path.R=numpy.zeros((path.ni,len(path.p[0].R),path.dim),'d')
    for i in range(1,path.ni+1):
      if(method=='min'):i=0
      path.R[i-1]=path.p[i].R.copy()
    # put force in big vector
      if(method=='min'):break
    path.F=numpy.zeros((path.ni,len(path.p[0].R),path.dim),'d')
    for i in range(1,path.ni+1):
      if(method=='min'):i=0
      path.F[i-1]=path.p[i].F.copy()
      if(method=='min'):break

    try: path.start
    except: path.start=0
    if(not path.start):
      path.start=1
      path.a=numpy.zeros(self.memory+1,'d')
      self.ptmp=copy.deepcopy(path)
      # set maxmove to give average maxmove perimage
      self.maxmove=numpy.sqrt(self.maxmove*path.ni)
      path.lbfgsinit=0
    try: path.lbfgsinit
    except: path.lbfgsinit=0
    if(not path.lbfgsinit):
      path.lbfgsinit=1
      path.Ho=numpy.ones((path.ni,len(path.p[0].R),path.dim),'d')
      if(not self.min=='line'): path.Ho=path.Ho*self.alpha
      path.ITR=1
      path.s=[1.]
      path.y=[1.]
      path.rho=[1.]
    else:
      a1=abs(vdot(path.F,path.Fold))
      a2=vdot(path.Fold,path.Fold)
 
      # check for reset
      if(self.min=='line'):
        if(a1<=0.5*a2 and a2!=0):
          reset_flag = 0
        else:
          reset_flag = 1
      else:
        reset_flag = 0

      if(reset_flag==0):
        ITR=path.ITR
        if(ITR>self.memory):
          path.s.pop(1)
          path.y.pop(1)
          path.rho.pop(1)
          ITR=self.memory
        path.s.append(path.R-path.Rold)
        # boundry cond
        for i in range(path.ni):
          if(method=='min'):i=0
          try:
            DBC(path.s[ITR][i],path.p[i].Box) #need to make matrix for box
          except:
            print("Box not found.")
          if(method=='min'):break
        path.y.append(-(path.F-path.Fold))
        path.rho.append(1/vdot(path.y[ITR],path.s[ITR]))
        path.ITR+=1
      else:
#        print 'reset image',i
        path.ITR=1
        path.s=[1.]
        path.y=[1.]
        path.rho=[1.]

    # compute Ho*G
    path.Rold=path.R.copy()
    path.Fold=path.F.copy()

    if(path.ITR<=self.memory):
      BOUND=path.ITR
    else:
      BOUND=self.memory
    
    q=-1.0*path.F

    for j in range(1,BOUND):
      k=(BOUND-j)
      path.a[k]=path.rho[k]*vdot(path.s[k],q)
      q-=path.a[k]*path.y[k]

    d=path.Ho*q # no needed cause Ho is idenity matrix

    for j in range(1,BOUND):
      B=path.rho[j]*vdot(path.y[j],d)
      d=d+path.s[j]*(path.a[j]-B)

    d=-1.0*d
    if(not self.min=='line'): path.d=d
    path.du=vunit(d)
    if(self.min=='line'):

      # Finite difference step using temporary point
      self.ptmp.R=path.R.copy()
      self.ptmp.Umaxi=path.Umaxi
#   must be turned on if tanset is commented
#        self.ptmp.p[i].N=path.p[i].N.copy()

      self.ptmp.R+=(path.du*self.dR)

      # put back in points
      for i in range(1,path.ni+1):
        if(method=='min'):i=0
        self.ptmp.p[i].R=self.ptmp.R[i-1]
        force(self.ptmp.p[i])
        if(method=='min'):break

    else:
    # use the Hessian Matrix to predict the min
      if(abs(vmag(path.d))>self.maxmove):
        path.d=path.du*self.maxmove
        print("maxmove", end=' ')
      path.R+=path.d

    if(self.min=='line'):
     # force projections on a small stepa
      if(method=='neb' or method=='str' ):
        self.ptmp.tangentset()# if commented copy N above
        self.ptmp.project()
      # put force in big vector
      self.ptmp.F=numpy.zeros((path.ni,len(path.p[0].F),path.dim),'d')
      for i in range(1,path.ni+1):
        if(method=='min'):i=0
        self.ptmp.F[i-1]=self.ptmp.p[i].F
        if(method=='min'):break

      # Decide how much to move along the line du
      Fp1=vdot(path.F,path.du)
      Fp2=vdot(self.ptmp.F,path.du)
      CR=(Fp1-Fp2)/self.dR
      if(CR<0.0):
        print("negcurve")
        RdR=self.maxmove
#        RdR=Fp1*0.1
        if(abs(RdR)>self.maxmove):
          RdR=sign(RdR)*self.maxmove
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

    # put back in points
    for i in range(1,path.ni+1):
      if(method=='min'):i=0
      path.p[i].R=path.R[i-1]
      if(method=='min'):break

##################################################
##################################################
