##################################################
#  Lennard-Jones potential
#
#  Version 1
#  Last modified by Dan 2006.09.09
##################################################
import numpy
import math
import lj_

##################################################
# Global Variables
##################################################
#Epsilon=1.0        # Epsilon: well depth
#Sigma=1.0          # Sigma: bond length parameter
#RC=5.0             # RC: Cutoff distance
##################################################
# Derived Global Variables
##################################################
#U0=4.0*Epsilon
#UC=U0*(math.pow(Sigma/RC,12)-math.pow(Sigma/RC,6))
#RCsq=RC*RC

##################################################
# Initialize Parameters
##################################################
def init(epsilon=1.0, sigma=1.0, rc=10.0):
  global Epsilon, Sigma, RC, U0, UC, RCsq
  Sigma=sigma
  Epsilon=epsilon
  RC=rc
  U0=4.0*Epsilon
  UC=U0*(math.pow(Sigma/RC,12)-math.pow(Sigma/RC,6))
  RCsq=RC*RC
  PotInitFlag=1
###############################################
# Wrapper to fortran lennard-jones potential
###############################################
def force(p):
  global Sigma, Epsilon, RC, U0, UC, RCsq
  ncoo=p.NumAtoms*3
  Rflat=p.R.flat
  ra=numpy.zeros(ncoo, 'd')
  fa=numpy.zeros(ncoo, 'd')
  uRet=numpy.array([0],'d')
  for i in range(ncoo):
    ra[i]=Rflat[i]
  ax=p.Box[0][0]
  ay=p.Box[1][1]
  az=p.Box[2][2]
  lj_.force(ra,fa,uRet,ax,ay,az,U0,Epsilon,Sigma,RC)

  p.F=numpy.resize(fa,(p.NumAtoms,3))
#  print 'FORCE:  ',p.F
#  p.F*=p.FreeD  # Set frozen atoms to have no force
  try: p.F*=p.FreeD
  except: p.FreeD=numpy.ones((p.NumAtoms,p.dim),'d')
  p.U=uRet[0]

