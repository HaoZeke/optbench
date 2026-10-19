##################################################
import copy
import numpy
import numpy.linalg as la
from Gman.func import vdot, vunit,vmag,DBC
from Gman.math import sign
class newton:
##################################################
# newton initialization
##################################################
  def init(self,maxmove=0.2,alpha=5,min='global',graphic=None):
    """Newton initializer function, called in script before neb
    def init(self,maxmove=0.2,dR=0.0001,memory=4):
    Optional arguments:
      maxmove:  maximum amount the point can move during optimization
      alpha:  spring constant """

    print('newton optimizer parameters:')
    print('  maxmove                  = ',maxmove)
    print('  alpha                    = ',alpha)
    print('  min                      = ',min)
    self.maxmove=maxmove
    self.alpha=alpha
    self.min=min
    self.graphic=graphic

##################################################
# NEB: newton step for NEB
##################################################
  def step(self,path,force,method='min'):
    """newton step, only called from neb
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
      # set maxmove to give average maxmove per image
      self.maxmove=numpy.sqrt(self.maxmove*path.ni)
      path.newton=0
    try: path.newton
    except: path.newton=0
    if(not path.newton):
#    force(self.p[i])
      path.newton=1
      path.size=len(path.p[0].R)*path.ni
      path.H=numpy.identity(path.dim*path.size,'d')
      path.H=path.H*2.0*self.alpha

#     global or local hessian matrix
      if(self.min=='global'):
        for i in range(path.size):
          for j in range(path.size):
            if(i==j+path.dim or i==j-path.dim): path.H[i][j]=-self.alpha
      path.Bo=la.inv(path.H)

#     global or local inv-hessian matrix
#      for i in range(path.size):
#        for j in range(path.size):
#          if(i==j+path.dim or i==j-path.dim): path.H[i][j]=-self.alpha
#      path.Bo=la.inv(path.H)
#      if(not self.min=='global'):
#        for i in range(path.size):
#          for j in range(path.size):
#            if(j>=i+path.dim or j<=i-path.dim): path.Bo[i][j]=0.0




    if(not self.graphic==None):
      self.graphic()

    path.Gold = numpy.reshape(-path.F,(path.dim*path.size,1))
    path.d = numpy.dot(-path.Bo,path.Gold)
    path.d3 = numpy.reshape(path.d,numpy.shape(self.ptmp.R))
    path.du = vunit(path.d3)

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
