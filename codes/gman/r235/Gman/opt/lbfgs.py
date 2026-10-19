##################################################
import copy
import numpy
from Gman.func import vdot,vunit,vmag,DBC
from Gman.math import sign
class lbfgs:
##################################################
# NEB: LBFGS initialization
##################################################
  def init(self,maxmove=0.2,dR=0.0001,memory=25,min='line',alpha=0.05):
    """Limited memory bfgs initializer function, called in script before neb
    def init(self,maxmove=0.2,dR=0.0001,memory=4,min='line'):
    Optional arguments:
      dR:       finite difference step size
      maxmove:  maximum amount the point can move during optimization
      memory:   number of preceding steps remembered"""
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
      path.CI=0.
      path.NEXT=0.
      path.TWO_NEXT=0.
      path.OTHER=0.
      path.start=1
      self.ptmp=copy.deepcopy(path)
      path.a=numpy.zeros(self.memory+1,'d')
      path.size=len(path.p[0].R)
      for i in range(1,path.ni+1):
        if(method=='min'):i=0
        path.p[i].lbfgsinit=0
        if(method=='min'):break
    for i in range(1,path.ni+1):
      if(method=='min'):i=0
      try: path.p[i].lbfgsinit
      except: path.p[i].lbfgsinit=0
      if(not path.p[i].lbfgsinit):
        path.p[i].lbfgsinit=1
        path.p[i].Ho=numpy.ones(path.size*path.dim,'d')
        if(not self.min=='line'): path.p[i].Ho=path.p[i].Ho*self.alpha
        path.p[i].Ho=numpy.reshape(path.p[i].Ho,(path.size,path.dim))
        path.p[i].ITR=1
        path.p[i].s=[1.]
        path.p[i].y=[1.]
        path.p[i].rho=[1.]
      else:
        a1=abs(vdot(path.p[i].F,path.p[i].Fold))
        a2=vdot(path.p[i].Fold,path.p[i].Fold)
 
        # check for reset
        if(self.min=='line'):
          if(a1<=0.5*a2 and a2!=0):
            reset_flag = 0
          else:
            reset_flag = 1
        else:
          reset_flag = 0

        if(reset_flag==0):
          ITR=path.p[i].ITR
          if(ITR>self.memory):
            path.p[i].s.pop(1)
            path.p[i].y.pop(1)
            path.p[i].rho.pop(1)
            ITR=self.memory
          path.p[i].s.append(path.p[i].R-path.p[i].Rold)

          # boundary conditions
          try:
            DBC(path.p[i].s[ITR],path.p[i].Box)
          except:
            pass
          path.p[i].y.append(-(path.p[i].F-path.p[i].Fold))
          Invrho=vdot(path.p[i].y[ITR],path.p[i].s[ITR])
          if(Invrho==0.0): 
            path.start = 0
            break
          path.p[i].rho.append(1./Invrho)
          path.p[i].ITR+=1
        else:
#          print 'reset image',i
          path.p[i].ITR=1
          path.p[i].s=[1.]
          path.p[i].y=[1.]
          path.p[i].rho=[1.]

     # compute Ho*G
      path.p[i].Rold=path.p[i].R.copy()
      path.p[i].Fold=path.p[i].F.copy()

      if(path.p[i].ITR<=self.memory):
        BOUND=path.p[i].ITR
      else:
        BOUND=self.memory
    
      q=-1.0*path.p[i].F

      for j in range(1,BOUND):
        k=(BOUND-j)
        path.a[k]=path.p[i].rho[k]*vdot(path.p[i].s[k],q)
        q-=path.a[k]*path.p[i].y[k]

      d=path.p[i].Ho*q 

      for j in range(1,BOUND):
        B=path.p[i].rho[j]*vdot(path.p[i].y[j],d)
        d=d+path.p[i].s[j]*(path.a[j]-B)

      d=-1.0*d
      if(not self.min=='line'): path.p[i].d=d
      path.p[i].du=vunit(d)
      if(self.min=='line'):

        # finite difference step using a temporary point
        self.ptmp.p[i].R=path.p[i].R.copy()
        self.ptmp.Umaxi=path.Umaxi
#   must be turned on if tanset is commented
#        self.ptmp.p[i].N=path.p[i].N.copy()

        self.ptmp.p[i].R+=(path.p[i].du*self.dR)
        force(self.ptmp.p[i])

      else:
    # use the Hessian matrix to predict the minimum
        if(abs(vmag(path.p[i].d))>self.maxmove):
          path.p[i].d=path.p[i].du*self.maxmove
          print("maxmove")
        path.p[i].R+=path.p[i].d
      if(method=='min'):break

    if(self.min=='line'):
       # force projections on a small step
      if(method=='neb' or method=='str' ):
        self.ptmp.tangentset() # if commented copy N above
        self.ptmp.project()
      # decide how much to move along the line du
      for i in range(1,path.ni+1):
        if(method=='min'):i=0
        Fp1=vdot(path.p[i].F,path.p[i].du)
        Fp2=vdot(self.ptmp.p[i].F,path.p[i].du)
        CR=(Fp1-Fp2)/self.dR
        if(CR==0.0):CR=-1.0
        if(CR<0.0):
#          print "neg curve"
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
#          RdR=Fp1*0.1
          if(abs(RdR)>self.maxmove):
            RdR=sign(RdR)*self.maxmove
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


##################################################
##################################################

