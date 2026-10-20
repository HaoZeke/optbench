##################################################
#  Lennard-Jones potential
#  Last modified by GH: Oct 30, 2006
##################################################
import numpy
import math

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
def init(epsilon=1.0, sigma=1.0, rc=5.0):
#def init()
  global Epsilon, Sigma, RC, U0, UC, RCsq
  Sigma=sigma
  Epsilon=epsilon
  RC=rc
  U0=4.0*Epsilon
  UC=U0*(math.pow(Sigma/RC,12)-math.pow(Sigma/RC,6))
  RCsq=RC*RC

##################################################
# Lennard-Jones Potential
##################################################
def LJ(Rij):
  global Epsilon, Sigma, RC, U0, UC
  Rsq=sum(Rij*Rij)
  if(Rsq>=RCsq):
    return 0, numpy.array([0, 0, 0])
  Fij=Rij.copy()
  iRsq=1.0/Rsq
  iRs2=Sigma*Sigma*iRsq
  iRs6=iRs2*iRs2*iRs2
  iRs12=iRs6*iRs6
  Fij=Rij*U0*(12.0*iRs12-6.0*iRs6)*iRsq
  print('UCUT: ', UC)
  return U0*(iRs12-iRs6)-UC,Fij

##################################################
# Bulk Lennard-Jones Potential
##################################################
def force(p):
  global Epsilon, Sigma, RC, U0, UC
  p.U=0.0
  p.F=p.R.copy()*0.0
  Fij=numpy.array([0,0,0],'d')
  Box=numpy.array([p.Box[0][0],p.Box[1][1],p.Box[2][2]],'d')
#  hBox=Box/2.0
  for ni in range(p.NumAtoms):
    for nj in range(ni):
#      Rij=p.R[3*ni:3*ni+3]-p.R[3*nj:3*nj+3]
      Rij=p.R[ni]-p.R[nj]
#      print "before",Rij

      for k in range(3):
        Rij[k]-=Box[k]*round(Rij[k]/Box[k])

#      print "after",Rij

#      for k in range(3):
#        while(Rij[k]<-hBox[k]):
#          Rij[k]+=Box[k]
#        while(Rij[k]>hBox[k]):
#          Rij[k]-=Box[k]

#      if(not(abs(Rij[0])>RC or abs(Rij[1])>RC or abs(Rij[2])>RC)):
#      if(not(Rij[0]>RC or  Rij[1]>RC or  Rij[2]>RC or \
#             Rij[0]<-RC or Rij[1]<-RC or Rij[2]<-RC)):
      Rsq=sum(Rij*Rij)
      if(Rsq<=RCsq):
        iRsq=1.0/Rsq
        iRs2=Sigma*Sigma*iRsq
        iRs6=iRs2*iRs2*iRs2
        iRs12=iRs6*iRs6
        Fij=Rij*U0*(12.0*iRs12-6.0*iRs6)*iRsq
        p.U+=U0*(iRs12-iRs6)-UC
        p.F[ni]+=Fij
        p.F[nj]-=Fij
#  p.F*=p.FreeD
  try: p.F*=p.FreeD
  except: p.FreeD=numpy.ones((p.NumAtoms,p.dim),'d')

