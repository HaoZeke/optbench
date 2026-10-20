##################################################
# Module for minimizing a configuration
##################################################
import copy
from Gman.func import vmag
from Gman.data import data
from Gman.opt import qm
import Gman.io as Gio

class min:
##################################################
# Minimize
##################################################
  def minimize(self,p,force,optf=qm.step,itrmax=100,fmax=0.001,ppd=1,converge='Fmag',movie=False,outfunc=Gio.savexyz,filename='movie.xyz'):
    """Optimize a point with the conjugate gradients routine
    def minimize(self,p,force,optf=qm.step,itrmax=100,fmax=0.001,ppd=1):
    Manditory arguments:
      p:        data point containing position p.R
      force:    potential function which sets p.U and p.F when force(p) is called
    Optional arguments:
      itrmax:   maximum number of iterations
      fmax:     stopping criteria for the force
      converge: Fmag -> convergence based on total force
                comp -> convergence based on largest force component 
      ppd:      print period; print every ppd iterations"""

    print('Minimization parameters:')
    print('  Convergence criteria     = ', fmax)
    print('  Convergence based on     = ', converge)
    print("MIN: entering optimization loop\n")

    if(converge!='Fmag') & (converge!='comp'):
      print('FATAL ERROR: convergence is based on nothing! must set converge= "Fmag" or "comp"')
      raise SystemExit

    self.p=[data.point()]*1
    self.p[0]=copy.deepcopy(p)
    self.ni=0
    try: open(filename,'w')
    except: "MIN: Can't open file ",filename
    if(movie): outfunc(self.p[0],filename);

    converged=False
    try: self.p[0].F
    except: force(self.p[0])
    prln=" %10s  %14s %14s %14s" % ("Iteration ","Energy   ","Total Force","Max Force ")
    print(prln)
    print("---------------------------------------------------------")
    for i in range(itrmax):
      if((i)%ppd==0):
#        Rdiff=0
#        for j in range (2,len(p.R)):
#          Rdiff+= abs(vmag(p.R[j]-p.R[1]))
        prln="%7d    %14.5f %14.5f %14.5f   " % (i,self.p[0].U,vmag(self.p[0].F),(abs(self.p[0].F)).max())
        print(prln, end=' ')
#        print "Itr: ",i," U: ",p.U," Ftot: ",vmag(p.F)," F: ",(abs(p.F)).max(),"Rdiff: ",Rdiff
        if(movie): outfunc(self.p[0],filename,'a');
      if(converge=='comp'):
        if((abs(self.p[0].F)).max()<fmax):
          converged=True
          break
      if(converge=='Fmag'):
        if(vmag(self.p[0].F)<fmax):
          converged=True
          break
      optf(self,force,method='min')
      force(self.p[0])
      print()
    print("\n---------------------------------------------------------\n")
    if(not converged):  print("Minimizer did not converge after",itrmax,"iterations")
    if(converged):
      print("Configuration minimized in",i,"iterations")
      prln="%s %d, %s %.5f, %s %.5f, %s %.5f" % ("Itr:",i,"U:",self.p[0].U,"Ftot:",vmag(self.p[0].F),"Fmax:",(abs(self.p[0].F)).max())
      print(prln)
#      print "Itr: ",i," U: ",p.U," Ftot: ",vmag(p.F)," F: ",(abs(p.F)).max()
      if(movie): outfunc(self.p[0],filename,'a');
#    p=copy.deepcopy(self.p[0])
    p.R=self.p[0].R
    p.F=self.p[0].F
    p.U=self.p[0].U
#    print p.r
##################################################
