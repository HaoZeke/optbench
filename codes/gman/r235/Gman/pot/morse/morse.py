##################################################
#  Morse potential
#
#  Version 3
#  Last modified by GH 2004.07.08
##################################################
import numpy
from . import morse_

##################################################
# Global Variables
##################################################
U0=0.7102          # U0: well depth
R0=2.8970          # R0: diatomic distance
Alpha=1.6047       # Alpha: stiffness of potential
RC=9.5             # RC: Cutoff distance

##################################################
# Derived Global Variables
##################################################
UC=U0*(1-numpy.exp(Alpha*(R0-RC)))*(1-numpy.exp(Alpha*(R0-RC)))
RCsq=RC*RC
PotInitFlag=0

##################################################
# Initialize Parameters
##################################################
def init(u0 = U0, alpha = Alpha, r0 = R0, rc = RC):
  global U0, Alpha, R0, PotInitFlag, RC, UC, RCsq
  U0 = u0
  Alpha = alpha
  R0 = r0
  RC = rc
  UC = U0 * (1 - numpy.exp(Alpha * (R0 - RC))) * (1 - numpy.exp(Alpha * (R0 - RC)))
  RCsq = RC * RC
  PotInitFlag = 1

###############################################
# Wrapper to fortran morse potential
###############################################
def force(p):
  global U0, Alpha, R0, PotInitFlag,NumFC, RC, UC, RCsq
  ncoo = p.NumAtoms * 3
  #print numpy.shape(p.R)
  p.F = 0.0 * p.R.copy()
  Rflat = p.R.flat
  ra = numpy.zeros(ncoo,'d')
  fa = numpy.zeros(ncoo,'d')
  #for i in range(ncoo):
  #  ra[i] = Rflat[i]
  #uRet=numpy.array([0.0])
  #ax=p.Box[0][0]
  #ay=p.Box[1][1]
  #az=p.Box[2][2]

  for i in range(p.NumAtoms):
    for j in range(p.dim):
      ra[3 * i + j] = Rflat[p.dim * i + j]
  uRet = numpy.array([0.0])
  try: ax = p.Box[0][0]
  except: ax = 100
  try: ay = p.Box[1][1]
  except: ay = 100
  try: az = p.Box[2][2]
  except: az = 100

  morse_.force(ra,fa,uRet,ax,ay,az,U0,Alpha,R0,RC)

  for i in range(p.NumAtoms):
    for j in range(p.dim):
      p.F[i][j] = fa[3 * i + j]

  #p.F=numpy.resize(fa,(p.NumAtoms,3))
  p.F *= p.FreeD  # Set frozen atoms to have no force
#  try: p.F*=p.FreeD
#  except: p.FreeD=numpy.ones((p.NumAtoms,p.dim),'d')
  p.U = uRet[0]

