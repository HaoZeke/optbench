##################################################
#  Lennard-Jones potential for two types of atoms
#
#  Version 1
#  Last modified by Dan 2010.05.05
##################################################
import numpy
import math
import lj_

##################################################
# Global Variables 
# Peng, X.Q. Composites: Part A 36 (2005) 859-874
##################################################
#EpsilonAA=1.14        # Epsilon: well depth
#EpsilonBB=1.0        # Epsilon: well depth
#EpsilonAB=0.223        # Epsilon: well depth
#SigmaAA=0.9          # Sigma: bond length parameter
#SigmaBB=0.865          # Sigma: bond length parameter
#SigmaAB=0.6          # Sigma: bond length parameter
#RC=10.0             # RC: Cutoff distance
##################################################
# Derived Global Variables
##################################################
#U0=4.0*Epsilon
#UC=U0*(math.pow(Sigma/RC,12)-math.pow(Sigma/RC,6))
#RCsq=RC*RC

##################################################
# Initialize Parameters
##################################################
def init(epsilonAA=1.14, sigmaAA=0.9,epsilonBB=1.0, sigmaBB=0.865,epsilonAB=0.223, sigmaAB=0.6, rc=10.0):
#def init(epsilonAA=1.0, sigmaAA=1.0,epsilonBB=1.0, sigmaBB=1.0,epsilonAB=1.0, sigmaAB=1.0, rc=10.0):
  global RC,RCsq,EpsilonAA, SigmaAA, U0_AA, UC_AA, RCsq,EpsilonBB, SigmaBB, U0_BB, UC_BB,EpsilonAB, SigmaAB, U0_AB, UC_AB
  SigmaAA=sigmaAA
  SigmaBB=sigmaBB
  SigmaAB=sigmaAB
  EpsilonAA=epsilonAA
  EpsilonBB=epsilonBB
  EpsilonAB=epsilonAB
  RC=rc
  U0_AA=4.0*EpsilonAA
  UC_AA=U0_AA*(math.pow(SigmaAA/RC,12)-math.pow(SigmaAA/RC,6))
  U0_BB=4.0*EpsilonAA
  UC_BB=U0_BB*(math.pow(SigmaBB/RC,12)-math.pow(SigmaBB/RC,6))
  U0_AB=4.0*EpsilonAB
  UC_AB=U0_AB*(math.pow(SigmaAA/RC,12)-math.pow(SigmaAB/RC,6))
  RCsq=RC*RC
  PotInitFlag=1
###############################################
# Wrapper to fortran lennard-jones potential
###############################################
def force(p):
  global RC,RCsq,EpsilonAA, SigmaAA, U0_AA, UC_AA, RCsq,EpsilonBB, SigmaBB, U0_BB, UC_BB,EpsilonAB, SigmaAB, U0_AB, UC_AB
  NA=p.NumAtom[0]
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
  lj_.force(ra,fa,uRet,ax,ay,az,U0_AA,SigmaAA,U0_BB,SigmaBB,U0_AB,SigmaAB,RC,NA)

  p.F=numpy.resize(fa,(p.NumAtoms,3))
#  print 'FORCE:  ',p.F
#  p.F*=p.FreeD  # Set frozen atoms to have no force
  try: p.F*=p.FreeD
  except: p.FreeD=numpy.ones((p.NumAtoms,p.dim),'d')
  p.U=uRet[0]

