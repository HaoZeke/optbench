import copy
import numpy
from   Gman.data                    import data
from   Gman.func                    import vmag, vproj, vdot, vunit
from   math                         import sqrt
from   Gman.opt                     import qm
from   Gman.data.ParametricSplineND import *


class newString: 


####################################################################################################################################################
# STRING: new linear band
#    new: must be called in script to create the string
#    def new(self,p1,p2,images=8):
#    Mandatory arguments:
#      p1:       initial configuration
#      p2:       final configuration
#    Optional arguments:
#      images:   number of images along the neb (excluding endpoints)
####################################################################################################################################################

  def new(self, p1, p2, images = 8):
    self.k = 5
    self.ni = images
    try: p1.dim
    except: p1.dim=numpy.shape[1]
    self.dimensions = p1.dim
    self.p = [data.point()] * (self.ni + 2)
    self.p[0] = p1
    self.p[self.ni + 1] = p2
    dR=(p2.R - p1.R) / (self.ni + 1.0)
    for i in range(1, self.ni + 1):
      self.p[i] = copy.deepcopy(p1)
      self.p[i].R += dR * (i * 1.0)
    self.init = False
    self.hasMoved = False
    self.oldPoints = copy.deepcopy(self.p)


####################################################################################################################################################
# Spline functions
####################################################################################################################################################

  def getElements(self, dimension):
    z = list(range(self.ni + 2))
    for i in range(self.ni + 2):
      temp = self.p[i].R.flat
      z[i] = temp[dimension]
    return z

  def getPointLength(self, point):
    total = 0
    for i in range(point):
      total += vmag(self.p[i + 1].R - self.p[i].R)
    return total

  def getPointLengths(self):
    z = list(range(self.ni + 2))
    total = 0
    for i in range(self.ni + 2):
      z[i] = total
      if i < self.ni + 1:
        total += vmag(self.p[i + 1].R - self.p[i].R)
    return z

  def getLength(self):
    total = 0
    for i in range(self.ni + 1):
      total += vmag(self.p[i + 1].R - self.p[i].R)
    return total

  def generateSpline(self):
    self.spline = ParametricSplineND()
    z = self.getPointLengths()
    for i in range(self.p[1].dim * self.p[1].NumAtom):
      self.spline.addDimension(z, self.getElements(i))

  def getSplineValue(self, t):
    return self.spline.pointAtLength(t)      
    

####################################################################################################################################################
# STRING: relax
#    relax the string with the quick-min routine
#    def relax(self,force,optf=qmstep,itrmax=100,fmax=0.001,tangent='new',method='off',ppd=1,fsaddle='off'):
#    Mandatory arguments:
#      force:    potential function which sets p.U and p.F when force(p) is called
#    Optional arguments:
#      optf:     optimization step function, which must implement optf(p,force)
#      itrmax:   maximum number of iterations
#      fmax:     stopping criteria for the force
#      converge: 'Fmag' convergence based on the magnitude of the Force
#                'comp' convergence based on the largest component of the Force
#      method:   'ci' this turns on the climbing image
#      fsaddle:  if 'on' the stopping criteria is only based on the climbing image
#      tangent:  'new' tangent is based upon the highest energy neighbor
#                'old' uses a mid-point scheme to find the local tangent
####################################################################################################################################################

  def relax(self, force, optf = qm.step, itrmax = 100, fmax = 0.001, tangent = 'new', method = 'off', ppd = 1, fsaddle = 'off', converge = 'distance', graphic = None, RMSmin = 0.001):
    print('')
    print("String Parameters:")
    print('  Convergence criteria     = ', fmax)
    print('  Convergence based on     = ', converge)
    print('  Tangent                  = ', tangent)
    print('  Climbing image method    = ', method)
    print('  Convergence based on ci  = ', fsaddle)
    print('')
    print("STRING: entering optimization loop")
    print('')
    self.converged = False
    self.springConverged = False
    self.method = method
    self.tangent = tangent
    if(fsaddle == 'on') & (method != 'ci'):
      print('WARNING: Climbing image not set to "on". Stopping criteria is based upon the force of all images') 
    if(converge != 'Fmag') & (converge != 'distance'):
      print('FATAL ERROR: convergence is based on nothing; must set converge= "Fmag" or "distance"')
      raise SystemExit 

# Do a force call on each image if the STRING has not been initialized
    if(not self.init):
      force(self.p[0], False)
      force(self.p[self.ni+1], False)
      for i in range(self.ni+2):
        self.p[i].V=self.p[i].R*0.0
      self.init=True

# Set up counters for neg curv
    self.CI = 0
    self.NEXT = 0
    self.TWO_NEXT = 0
    self.OTHER = 0

# Main relax loop.
    for itr in range(itrmax):

# Evaluate the energy and force
      self.Umax = self.p[0].U
      self.Umaxi = 0
      for i in range(1, self.ni + 1):
        force(self.p[i])
        if(self.p[i].U > self.Umax):
          self.Umax = self.p[i].U
          self.Umaxi = i

#Check for convergence
      if self.RMSdistance() < RMSmin and self.hasMoved:
        self.converged = True
      if(converge == 'Fmag'):
        fci = vmag(self.p[self.Umaxi].F)
        fmaxcur = vmag(self.p[1].F)
        for i in range(2, self.ni + 1):
          fmaxcur = max(fmaxcur, vmag(self.p[i].F))
        if(fsaddle == 'on') and (method == 'ci') and (fci < fmax):
          self.converged = True
          print('Climbing Image: converged')
        if(fmaxcur < fmax) or (self.converged == True):
          self.converged = True
          print('STRING: converged in ', itr, ' iterations.')
        if(fmaxcur < fmax) and (reset == True) and (method == 'ci'):
          self.redistributeClimbingImage()
          self.start = 0
        elif(fmaxcur < fmax) and (reset == True):
          self.redist()
          self.start = 0
      if(self.converged):
        print("%6i %12.6f %6i %14.6f" % (itr+1,self.Umax-self.p[0].U,self.Umaxi, self.RMSdistance()))
        self.itr = itr
        break
      else:
        if(itr == 0):
          print("Iteration      MaxU       MaxI       RMS distance")
          print("-------------------------------------------------")
        if((itr + 1) % ppd == 0):
          print("%6i %12.6f %6i %14.6f" % (itr+1,self.Umax-self.p[0].U,self.Umaxi, self.RMSdistance()))
        if(not graphic == None): 
          graphic()

# Optimization call
      #self.tangentset()
      self.oldPoints = copy.deepcopy(self.p)
      optf(self, force, 'newstr')
      self.hasMoved = True
      if self.method == "ci":
          self.redistributeClimbingImage()
      else:
          self.redist()
    #if(self.converged):
    #  self.extfind(force)


####################################################################################################################################################
# Redistributing the points
####################################################################################################################################################

  def redist(self):
    self.generateSpline()
    length = self.getLength() / float(self.ni + 1)
    for i in range(1, self.ni + 1):
      self.p[i].Rtmp = self.getSplineValue(float(i) * length)
    for i in range(1, self.ni + 1):
      self.p[i].R = numpy.array([self.p[i].Rtmp])
      self.p[i].R = self.p[i].R.reshape(self.p[i].NumAtoms, self.p[i].dim)
     

  def pointatlen(self,number):
    total = 0.
    i = 0
    while(vmag(self.p[i + 1].R - self.p[i].R)+total < number):
      total += vmag(self.p[i + 1].R - self.p[i].R)
      i = i + 1
    d1 = (number - total) * vunit(self.p[i + 1].R - self.p[i].R)
    return self.p[i].R + d1

####################################################################################################################################################
# Redistributing the points for the climbing image
####################################################################################################################################################

  def redistributeClimbingImage(self):
    self.lenLeft()
    self.dmag /= float(self.Umaxi)
    j = 1.0
    for i in range(self.Umaxi - 1, 0, -1):
      self.p[i].Rtmp = self.pointAtLenLeft(j * self.dmag)
      j += 1.0
    for i in range(self.Umaxi - 1, 0, -1):
      self.p[i].R = self.p[i].Rtmp.copy()
    self.lenRight()
    self.dmag /= float(self.ni + 1 - self.Umaxi)
    j = 1.0
    for i in range(self.Umaxi + 1, self.ni + 1):
      self.p[i].Rtmp = self.pointAtLenRight(j * self.dmag)
      j += 1.0
    for i in range(self.Umaxi + 1, self.ni + 1):
      self.p[i].R = self.p[i].Rtmp.copy()

  def lenLeft(self):
    self.dmag = 0.0
    for i in range(self.Umaxi, 0, -1):
      d1 = self.p[i - 1].R - self.p[i].R
      self.dmag += vmag(d1)

  def lenRight(self):
    self.dmag = 0.0
    for i in range(self.Umaxi, self.ni + 1):
      d1 = self.p[i + 1].R - self.p[i].R
      self.dmag += vmag(d1)

  def pointAtLenLeft(self, length):
    total = 0.0
    i = self.Umaxi
    while(vmag(self.p[i - 1].R - self.p[i].R) + total < length):
      total += vmag(self.p[i - 1].R - self.p[i].R)
      i = i - 1
    d1 = (length - total) * vunit(self.p[i - 1].R - self.p[i].R)
    return self.p[i].R + d1

  def pointAtLenRight(self, length):
    total = 0.0
    i = self.Umaxi
    while(vmag(self.p[i + 1].R - self.p[i].R) + total < length):
      total += vmag(self.p[i + 1].R - self.p[i].R)
      i = i + 1
    d1 = (length - total) * vunit(self.p[i + 1].R - self.p[i].R)
    return self.p[i].R + d1

####################################################################################################################################################
# String: find extrema
#    A function to find the extrema along the string after it has been relaxed
#    def extfind(self,force):
#    called from the string
####################################################################################################################################################

  def extfind(self,force):
    print('String: calculating extrema')
    for i in range(1,self.ni+1):
      force(self.p[i], False)
    self.NumExts=0
    # Get normals for the endpoints
    self.tangentset()
    # Set position, energy and force along the band
    self.MEP=[0.0]*(self.ni+2)
    self.MFP=[0.0]*(self.ni+2)
    self.MRP=[0.0]*(self.ni+2)
    # The cubic coefficients are define for the ni+1 intervals
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
    # Loop over each interval and find cubic coefficients
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
      self.Ext.sort(lambda x,y: cmp(x.R,y.R))
    for i in range(self.NumExt):
      print("  Extrema: %3i,  at point: %9.5f,  position: %9.5f,  with energy: %10.6f" % (i,self.Ext[i].X,self.Ext[i].R,self.Ext[i].U))
    print("String: done finding extrema")



####################################################################################################################################################
# STRING: write out
#    function to save an NEB to a file
#    def save(self,outfunc,outfile):
#    Required parameters:
#      outfunc:  write function, must implement outfunc(p,filename,w='a')
#      filename:  filename to write
####################################################################################################################################################

  def save(self,outfunc,filename):
    print("NEB: saving string")
    try:
      file=open(filename,'w')
    except:
      "NEB: Can't open file ",filename
      return
    for i in range(self.ni+1):
      outfunc(self.p[i],filename,w='a')
#      outfunc(self.p[i],filename+str(self.itr),w='a')
    print("NEB: done saving string")



####################################################################################################################################################
# STRING: read in
#   function to load NEB images from a file
#    def load(self,infunc,filename):
#    Required parameters:
#      infunc:  write function, must implement infunc(p,filename,[file])
#      filename:  filename to read
####################################################################################################################################################

  def load(self,infunc,filename):
    print("STRING: loading string from ",filename)
    try:
      file=open(filename,'r')
    except:
      "NEB: Can't open file ",filename
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

####################################################################################################################################################
# String: export path to Mathematica Line graphic
####################################################################################################################################################
  def toMathematica(self, name):
    outString = name + " = Graphics[{"
    outString = outString + "Line[{ "
    self.generateSpline()
    step = self.getLength() / 400
    t = 0
    p1 = self.getSplineValue(0)
    outString = outString + "{%f" % p1[0] + ", " + "%f" % p1[1] + "}, "
    while t < self.getLength() - step:
        p1 = self.getSplineValue(t + step)
        outString = outString + "{%f" % p1[0] + ", " + "%f" % p1[1] + "}, "
        t += step
    p1 = self.getSplineValue(self.getLength())
    outString = outString + "{%f" % p1[0] + ", " + "%f" % p1[1] + "} "
    outString = outString + "}], "
    for i in range(self.ni + 1):
      outString = outString + "Circle[{%f, %f" % (self.p[i].R[0][0], self.p[i].R[0][1]) + "}, circleSize], "
    outString = outString + "Circle[{%f, %f" % (self.p[self.ni + 1].R[0][0], self.p[self.ni + 1].R[0][1]) + "}, circleSize]"
    outString = outString + "}];"
    return outString;
  
####################################################################################################################################################
# STRING: Returns the RMS distance from the current iteration to the previous.
####################################################################################################################################################
  def RMSdistance(self):
    total = 0
    for i in range(1, self.ni + 1):
      total += self.oldPoints[i].distanceTo(self.p[i]) * self.oldPoints[i].distanceTo(self.p[i])
    total /= self.ni
    return sqrt(total)



####################################################################################################################################################
# STRING: shell object to hold extrema data
####################################################################################################################################################

class ext:
  def new(self):
    self.U=0.0
    self.X=0.0
    self.R=0.0
