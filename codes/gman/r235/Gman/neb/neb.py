##################################################
# Nudged elastic band routines
##################################################
import copy
from Gman.data import data
from Gman.func import vmag, vproj, vdot, vunit
from math import sqrt, atan, pi
from Gman.opt import qm

class neb: 
##################################################
# NEB: new linear band
##################################################
  def new(self,p1,p2,images=8):
    """NEB make: must be called in script to create the neb
    def new(self,p1,p2,images=8):
    Mandatory arguments:
      p1:       initial configuration
      p2:       final configuration
    Optional arguments:
      images:   number of images along the neb (excluding endpoints)"""
    self.ni=images
    self.p=[data.point()]*(self.ni+2)
    self.p[0]=p1
    self.p[self.ni+1]=p2
    dR=(p2.R-p1.R)/(self.ni+1.0)
    for i in range(1,self.ni+1):
      self.p[i]=copy.deepcopy(p1)
      self.p[i].R+=dR*(i*1.0)
#    force(self.p[i]) 
    self.oldPoints = copy.deepcopy(self.p)
    self.init=False
##################################################
# NEB: relax
##################################################
  def relax(self,force,optf=qm.step,itrmax=100,fmax=0.001,k=5.0,tangent='new',method='off',ppd=1,dneb='off',dneb_org='off',fsaddle='off',converge='Fmag',stats=False,graphic=None):
    """relax the neb with the quick-min routine
    def relax(self,force,optf=qmstep,itrmax=100,fmax=0.001,k=5.0,tangent='new',method='off',ppd=1,dneb='off',dneb_org='off',fsaddle='off'):
    Mandatory arguments:
      force:    potential function which sets p.U and p.F when force(p) is called
    Optional arguments:
      optf:     optimization step function, which must implement optf(p,force)
      itrmax:   maximum number of iterations
      fmax:     stopping criteria for the force
      converge: 'Fmag' convergence based on the magnitude of the Force
                'comp' convergence based on the largest component of the Force 

      method:   'ci' this turns on the climbing image
      fsaddle:  if 'on' the stopping criteria is only based on the climbing image
      tangent:  'new' tangent is based upon the highest energy neighbor
                'old' uses a mid-point scheme to find the local tangent
      ppd:      printing period; print output every ppd iterations
      dneb:     'on' or 'off' turns on double nudging
      dneb_org: 'on' this is the original double nudging, must also have dneb='on' for this to work"""
    print('')
    print("Nudged Elastic Band Parameters:")
    print('  Convergence criteria     = ', fmax)
    print('  Convergence based on     = ', converge)
    print('  Spring constant          = ', k)
    print('  Tangent                  = ', tangent)
    print('  Climbing image method    = ', method)
    print('  Convergence based on ci  = ', fsaddle)
    print('  Double nudging           = ', dneb)
    print('  Original double nudging  = ', dneb_org)
    print('')
    print("NEB: entering optimization loop")
    print('')
    self.ciconverged=False
    self.converged=False
    self.method=method
    self.tangent=tangent
    self.k=k
    self.dneb=dneb
    self.dneb_org=dneb_org
    if(fsaddle=='on') & (method!='ci'):
      print('WARNING: Climbing image not set to "on". Stopping criteria is based upon the force of all images') 
    if(dneb_org=='on') & (dneb=='on'):
      print('WARNING: the dneborg is "on"; the neb will probably not converge') 
    if(converge!='Fmag') & (converge!='comp'):
      print('FATAL ERROR: convergence is based on nothing; must set converge= "Fmag" or "comp"')
      raise SystemExit 
    # Do a force call on each image if the NEB has not been initialized
    if(not self.init):
      force(self.p[0],False)
      force(self.p[self.ni+1],False)
      for i in range(self.ni+2):
#        force(self.p[i])
        self.p[i].V=self.p[i].R*0.0
      self.init=True
    # Set up counters for neg curv
    if(stats):
      self.CI = 0
      self.NEXT = 0
      self.TWO_NEXT = 0
      self.OTHER = 0
    print("Iteration     Force        MaxU       MaxI    Force on CI    RMS Distance")
    print("-------------------------------------------------------------------------")
    # LOOP:
    for itr in range(itrmax):
      # Evaluate the energy and force
      self.Umax=self.p[0].U
      self.Umaxi=0
      for i in range(1,self.ni+1):
        force(self.p[i])
        if(self.p[i].U>self.Umax):
          self.Umax=self.p[i].U
          self.Umaxi=i
      # Set the tangent vectors
      #if itr == 0: self.tangentset()
      self.tangentset()
      # Force projections
      self.project()
      # image by image optimization
   #   dt=0.1
      #dt=0.0125
   #   for i in range(1,self.ni+1):
        #try: self.p[i].V
        #except: self.p[i].V=self.p[i].R.copy()*0.0
   #     try: self.V
   #     except: self.V=self.p[i].R.copy()*0.0
   #     for j in range(1):
   #       force(self.p[i],False)
   #       self.tangentset()
   #       self.project()
          # make sure force and velocity are in same direction (seems just as good)
          #if(vdot(self.p[i].F,self.p[i].V)>0):
          #  self.p[i].V=vproj(self.p[i].V,self.p[i].F)
          #else:
          #  self.p[i].V*=0.0
          ## Euler step 
          #dR=dt*self.p[i].V
          #self.p[i].V+=dt*vunit(self.p[i].F)
          ## check for max step
          ## move R
          #self.p[i].R+=dR

   #       if(vdot(self.p[i].F,self.V)>0):
   #         self.V=vproj(self.V,self.p[i].F)
   #       else:
   #         self.V*=0.0
          # Euler step 
   #       dR=dt*self.V
   #       self.V+=dt*vunit(self.p[i].F)
          # check for max step
          # move R
   #       self.p[i].R+=dR

      # Check for convergence
      if(converge=='comp'):
        fci=(abs(self.p[self.Umaxi].F)).max()
        self.fmaxcur=(abs(self.p[1].F)).max()
        for i in range(2,self.ni+1):
          self.fmaxcur=max(self.fmaxcur,(abs(self.p[i].F)).max())
      if(converge=='Fmag'):
        fci=vmag(self.p[self.Umaxi].F)
        self.fmaxcur = vmag(self.p[1].F)
        for i in range(2,self.ni+1):
          self.fmaxcur = max(self.fmaxcur,vmag(self.p[i].F))
      if(fsaddle=='on') & (method=='ci') & (fci<fmax):
        self.ciconverged=True
      if(self.fmaxcur<fmax) | (self.ciconverged):
        self.converged=True
      # Print data
      if(self.converged) | ((itr+1)%ppd==0):
        print("%6i %13.6f %12.6f %6i %14.6f %14.6f" % (itr+1,self.fmaxcur,self.Umax-self.p[0].U,self.Umaxi,fci, self.RMSdifference()))
      if(not graphic==None):
        graphic()
      if(self.converged):
        self.itr=itr
        break
      # Optimization call
      self.oldPoints = copy.deepcopy(self.p)
      optf(self,force,'neb')
    # END LOOP
    print("-------------------------------------------------------------------------")
    if(self.ciconverged):
      print('Climbing image: converged')
    if(self.converged):
      print('NEB: converged in ',itr,' Itr')
      self.extfind(force)
    else:
      print("NEB: finished, but did not converge")
    if(stats):
      print("NEB: NEG CURV COUNT")
      print("CI       =   ",self.CI)
      print("NEXT     =   ",self.NEXT)
      print("TWO_NEXT =   ",self.TWO_NEXT)
      print("OTHER    =   ",self.OTHER)
    print()
##################################################
# NEB: find tangent vector for each image 
##################################################
  def tangentset(self):
    """sets the local tangent based on neighboring images
    def tangentset(self):"""
    # Find tangent vector N
    self.p[0].N=self.p[1].R-self.p[0].R
    self.p[self.ni+1].N=self.p[self.ni+1].R-self.p[self.ni].R
    for i in range(1,self.ni+1):
      if(self.tangent=='old'):
        self.p[i].N=(self.p[i+1].R-self.p[i-1].R)
      else:
        UPm1=self.p[i-1].U>self.p[i].U
        UPp1=self.p[i+1].U>self.p[i].U
        if(UPm1!=UPp1):
          if(UPm1):
            self.p[i].N=self.p[i].R-self.p[i-1].R
          else:
            self.p[i].N=self.p[i+1].R-self.p[i].R
        else:
          Um1=self.p[i-1].U-self.p[i].U
          Up1=self.p[i+1].U-self.p[i].U
          Umin=min(abs(Up1),abs(Um1))
          Umax=max(abs(Up1),abs(Um1))
          if(Um1>Up1):
            self.p[i].N=(self.p[i+1].R-self.p[i].R)*Umin
            self.p[i].N+=(self.p[i].R-self.p[i-1].R)*Umax
          else:
            self.p[i].N=(self.p[i+1].R-self.p[i].R)*Umax
            self.p[i].N+=(self.p[i].R-self.p[i-1].R)*Umin
    for i in range(self.ni+2):
      self.p[i].N=vunit(self.p[i].N)
##################################################
# NEB: force projections
##################################################
  def project(self):
    """project out the real force along the tangent and add spring
    forces along the NEB to ensure that images are evenly spaced
    def project(self):"""
    for i in range(1,self.ni+1):
      if((self.method=='ci') & (i==self.Umaxi)):
        self.p[i].F-=2.0*vproj(self.p[i].F,self.p[i].N)
      else:
        # All images besides the climbing image
        self.p[i].Fperp=self.p[i].F-vproj(self.p[i].F,self.p[i].N)
        #self.p[i].Fperp=self.p[i].F+self.p[i].N*vdot(-self.p[i].F,self.p[i].N)
        Rm1=self.p[i-1].R-self.p[i].R
        Rp1=self.p[i+1].R-self.p[i].R
        self.p[i].FsN=(vmag(Rp1)-vmag(Rm1))*self.k*self.p[i].N
#        print i
#        print 'springs', abs(vmag(self.p[i].FsN))
#        print 'Fprep', abs(vmag(self.p[i].Fperp))

        if(self.dneb=='on'):
# for dneb use total spring force -spring force in the grad direction
          self.p[i].Fs=(Rp1+Rm1)*self.k
          self.p[i].Fsperp=self.p[i].Fs-self.k*vproj(Rp1+Rm1,self.p[i].N)  
          self.p[i].Fsdneb=self.p[i].Fsperp-vproj(self.p[i].Fs,self.p[i].Fperp)

# newdneb where dneb force converges with 
          if(self.dneb_org!='on'):
            FperpSQ=vdot(self.p[i].Fperp,self.p[i].Fperp)
            FsperpSQ=vdot(self.p[i].Fsperp,self.p[i].Fsperp)
            if(FsperpSQ>0):
              self.p[i].Fsdneb*=2./pi*atan(FperpSQ/FsperpSQ)
        else:
          self.p[i].Fsdneb=0
        self.p[i].F = self.p[i].Fsdneb+self.p[i].FsN+self.p[i].Fperp
        self.p[i].F*= self.p[i].FreeD
##################################################
# NEB: find extrema
##################################################
  def extfind(self,force):
    """ A function to find the extrema along the band after it has been relaxed
    def extfind(self,force):
    called from the neb"""
    print('NEB: calculating extrema')
    for i in range(1,self.ni+1):
      force(self.p[i],False)
    self.NumExts=0
    # Get normals for the endpoints
    self.tangentset()
    # Set position, energy and force along the band
    self.MEP=[0.0]*(self.ni+2)
    self.MFP=[0.0]*(self.ni+2)
    self.MRP=[0.0]*(self.ni+2)
    # The cubic coefficients are defined for the ni+1 intervals
    self.a=[0.0]*(self.ni+1)
    self.b=[0.0]*(self.ni+1)
    self.c=[0.0]*(self.ni+1)
    self.d=[0.0]*(self.ni+1)
    Uref=self.p[0].U
    for i in range(self.ni+2):
      self.MFP[i]=vdot(self.p[i].F,self.p[i].N)
      if(i==0):
        self.MRP[i]=0.0
        self.MEP[i]=0.0
      else:
        self.MRP[i]=self.MRP[i-1]+vmag(self.p[i].R-self.p[i-1].R)
        self.MEP[i]=self.p[i].U-Uref
    # Loop over each interval and find the cubic coefficients
    for i in range(self.ni-1):
      dr=self.MRP[i+1]-self.MRP[i]
      U1=self.MEP[i]
      U2=self.MEP[i+1]
      F1=self.MFP[i]*dr
      F2=self.MFP[i+1]*dr
      Ud=U2-U1
      Fs=F1+F2
      self.a[i]=U1
      self.b[i]=-F1
      self.c[i]=3.0*Ud+F1+Fs
      self.d[i]=-2.0*Ud-Fs
    # Find extrema
    self.NumExt=0
    self.Ext=[]
    for i in range(self.ni-1):
      desc=self.c[i]*self.c[i]-3*self.b[i]*self.d[i]
# New from vtstscripts
      if(desc>=0):
        f=-1;
        # Quadratic case  
        if((self.d[i]==0)&(self.c[i]!=0)):
          f=-(self.b[i]/(2*self.c[i]))
        # Cubic case 1
        elif(self.d[i]!=0):
          f=-(self.c[i]+sqrt(desc))/(3*self.d[i])
        if((f>=0)&(f<=1)):
          self.Ext.append(ext())
          self.Ext[self.NumExt].X=i+f
          self.Ext[self.NumExt].U=self.d[i]*f**3+self.c[i]*f**2+self.b[i]*f+self.a[i]
          self.Ext[self.NumExt].R=self.MRP[i]+f*(self.MRP[i+1]-self.MRP[i])
          self.NumExt+=1
        # Cubic case 2
        if(self.d[i]!=0):
          f=-(self.c[i]-sqrt(desc))/(3*self.d[i])
          if((f>=0)&(f<=1)):
            self.Ext.append(ext())
            self.Ext[self.NumExt].X=i+f
            self.Ext[self.NumExt].U=self.d[i]*f**3+self.c[i]*f**2+self.b[i]*f+self.a[i]
            self.Ext[self.NumExt].R=self.MRP[i]+f*(self.MRP[i+1]-self.MRP[i])
            self.NumExt+=1
    # Sort extrema based on distance
    if(self.NumExt>1):
      self.Ext.sort(key=lambda x: x.R)
    for i in range(self.NumExt):
      print("  Extrema: %3i,  at point: %9.5f,  position: %9.5f,  with energy: %10.6f" % (i,self.Ext[i].X,self.Ext[i].R,self.Ext[i].U))
    print("NEB: done finding extrema\n")
##################################################
# NEB: write out
##################################################
  def save(self,outfunc,filename):
    """function to save an NEB to a file
    def save(self,outfunc,outfile):
    Required parameters:
      outfunc:  write function, must implement outfunc(p,filename,w='a')
      filename:  filename to write"""
    print("NEB: saving neb")
    try: 
      open(filename,'w')
    except:
      "NEB: Can't open file ",filename
      return
    for i in range(self.ni+1):
      outfunc(self.p[i],filename,w='a')
#      outfunc(self.p[i],filename+str(self.itr),w='a')
    print("NEB: done saving neb")
##################################################
# NEB: read in
##################################################
  def load(self,infunc,filename):
    """function to load NEB images from a file
    def load(self,infunc,filename):
    Required parameters:
      infunc:  write function, must implement infunc(p,filename,[file])
      filename:  filename to read"""
    print("NEB: loading neb from ",filename)
    try:
      file=open(filename,'r')
    except:
      "NEB: Can not open file ",filename
      return
    Done=False
    self.ni=0
    self.p=[data.point()]
    pload=data.point()
    while (not Done):
#    for i in range(self.ni+1):
      try:
        infunc(pload,filename,file)
        self.p.append(data.point())
        self.p[self.ni]=copy.deepcopy(pload)
        self.ni+=1
      except:
        Done=True
    if(self.ni==0):
      print("NEB Error: failed to load any NEB images.")
      return
    elif(self.ni==1):
      print("NEB Error: only 1 image loaded, 3 are needed for the NEB.")
      return
    else:
      print("NEB: loaded ",self.ni-2,"(+2) NEB images")
    self.init=False
    self.ni-=2
    
##################################################
# NEB: export path to Mathematica Line graphic
##################################################
  def toMathematica(self, name, pointSize, showPoint=True):
    outString = name + " = Graphics[{"
    outString = outString + "Line[{ "
    for i in range(self.ni + 1):
      outString = outString + "{%f" % self.p[i].R[0][0] + ", " + "%f" % self.p[i].R[0][1] + "}, "
    outString = outString + "{%f" % self.p[self.ni + 1].R[0][0] + ", " + "%f" % self.p[self.ni + 1].R[0][1] + "} "
    outString = outString + "}]"
    if showPoint:
      outString = outString + ", "
      for i in range(self.ni + 1):
        outString = outString + "Disk[{%f, %f" % (self.p[i].R[0][0], self.p[i].R[0][1]) + "}, %f], " % pointSize
      outString = outString + "Disk[{%f, %f" % (self.p[self.ni + 1].R[0][0], self.p[self.ni + 1].R[0][1]) + "}, %f]" % pointSize
      outString = outString + "}];"
    return outString;
##################################################
# NEB: RMS distance between this path and the last path by image location.
##################################################
  def RMSdifference(self):
    total = 0
    for i in range(1, self.ni + 1):
      total += self.oldPoints[i].distanceTo(self.p[i]) * self.oldPoints[i].distanceTo(self.p[i])
    total /= self.ni
    return sqrt(total)

##################################################
# NEB: shell object to hold extrema data
##################################################
class ext:
  def new(self):
    self.U=0.0
    self.X=0.0
    self.R=0.0
