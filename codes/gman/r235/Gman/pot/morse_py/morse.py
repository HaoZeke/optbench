##################################################
#  Morse potential
#
#  Version 3
#  Last modified by GH Nov 14, 2004
##################################################

import numpy
import math

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

##################################################
# Initialize Parameters
# pass array of parameters (parms)
##################################################
#def init(u0=0.7102, alpha=1.6047, r0=2.8970, rc=5):
def init(u0=0.7102, alpha=1.6047, r0=2.8970, rc=5):
  global U0, Alpha, R0
  U0=u0
  Alpha=alpha
  R0=r0
  RC=rc
  UC=U0*(1-numpy.exp(Alpha*(R0-RC)))*(1-numpy.exp(Alpha*(R0-RC)))
  RCsq=RC*RC

##################################################
# Morse Potential
##################################################
def Morse(Rij):
  global U0, Alpha, R0
  Rsq=sum(Rij*Rij)
  if(Rsq>=RCsq):
    return 0, numpy.array([0, 0, 0])
  Fij=Rij.copy()
  R=sqrt(Rsq)
  Expr=_numpy.exp(Alpha*(R0-R))
  OmExpr=1.0-Expr
  tmp=(2*Alpha*U0*Expr*OmExpr/R)
  Fij=Rij*(2*Alpha*U0*Expr*OmExpr/R)
  return U0*OmExpr*OmExpr-UC,Fij

##################################################
# Bulk Morse Potential
##################################################
#def MorseBulk(p):
def force(p):
  global U0, Alpha, R0, RCsq
  p.U=0
  p.F=p.R.copy()*0.0
  Fij=numpy.array([0,0,0])
  Box=numpy.array([p.Box[0][0],p.Box[1][1],p.Box[2][2]])
  hBox=Box/2.0
  for ni in range(p.NumAtoms):
    for nj in range(ni):
#      Rij=p.R[3*ni:3*ni+3]-p.R[3*nj:3*nj+3]
      Rij=p.R[ni]-p.R[nj]
      for k in range(3):
        while(Rij[k]<-hBox[k]):
          Rij[k]+=Box[k]
        while(Rij[k]>hBox[k]):
          Rij[k]-=Box[k]
#      if(not(abs(Rij[0])>RC or abs(Rij[1])>RC or abs(Rij[2])>RC)):
      if(not(Rij[0]>RC or  Rij[1]>RC or  Rij[2]>RC or \
             Rij[0]<-RC or Rij[1]<-RC or Rij[2]<-RC)):
        Rsq=sum(Rij*Rij)
        if(Rsq<RCsq):
          Rs=math.sqrt(Rsq)
          Expr=numpy.exp(Alpha*(R0-Rs))
          OmExpr=1.0-Expr
          Fij=Rij*(2*Alpha*U0*Expr*OmExpr/Rs)
          p.U+=U0*OmExpr*OmExpr-UC
          p.F[ni]-=Fij
          p.F[nj]+=Fij
#          p.F[3*ni:3*ni+3]-=Fij[0:3]
#          p.F[3*nj:3*nj+3]+=Fij[0:3]
#  p.F*=p.FreeD
  try: p.F*=p.FreeD
  except: p.FreeD=numpy.ones((p.NumAtoms,p.dim),'d')

