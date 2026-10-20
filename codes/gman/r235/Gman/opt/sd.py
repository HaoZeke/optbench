##################################################
# Functions to do steepest descent step
##################################################
from Gman.func import vmag,vunit
class sd:
##################################################
# Steepest decent initialization
##################################################
  def init(self,maxmove=0.2,stepR=0.1):
    print('')
    print('Steepest-descent optimizer parameters:')
    print('  maxmove                  = ',maxmove)
    print('  stepR                    = ',stepR)
    self.maxmove=maxmove
    self.stepR=stepR
##################################################
# Steepest descent step
##################################################
  def step(self,path,force,method='min'):
    """
    Steepest-descent step
    def step(self,path,force):
    Mandatory arguments:
      path:     the entire neb run
      force:    potential function which sets p.U and p.F when force(p) is called
      method:   method which optimizer is called from
    """
    try: path.ni
    except: path.ni=0
    if(not path.ni):path.ni=1
    for i in range(1,path.ni+1):
      if(method=='min'):i=0 
      dR=path.p[i].F*self.stepR
      if(vmag(dR)>self.maxmove):
#        print "SD: MaxMove"
        dR=vunit(path.p[i].F)*self.maxmove
      path.p[i].R+=dR
      if(method=='min'):break 
##################################################
##################################################
