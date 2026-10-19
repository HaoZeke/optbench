##################################################
#  Morse potential
#
#  Version 3
#  Last modified by GH 2004.07.08
##################################################

from numarray import *
import string

##################################################
# File with Parameters
##################################################
ParmsFileName='PARMS_morse.dat'

##################################################
# Global Variables
##################################################
U0=0.7102          # U0: well depth
R0=2.8970          # R0: diatomic distance
Alpha=1.6047       # Alpha: stiffness of potential
RC=5               # RC: Cutoff distance
ParmsSetNum=0      # How many paramter sets have been loaded
ParmsLoadedFlag=0  # Have parameters been loaded?
ParmsData=''       # String to hold the parmater data sets

##################################################
# Derived Global Variables
##################################################
UC=U0*(1-exp(Alpha*(R0-RC)))*(1-exp(Alpha*(R0-RC)))
RCsq=RC*RC
PotInitFlag=0

##################################################
# Load paramters 
# (multiple calls for different parameters)
##################################################
def ParmsLoad():
  global ParmsLoadedFlag,ParmsFileName,ParmsData,ParmsSetNum
  Parms=array([0.0,0.0,0.0])
  if(not ParmsLoadedFlag):
    ParmsFile=open(ParmsFileName,'r')
    ParmsData=ParmsFile.readlines()
    ParmsFile.close
    ParmsLoadedFlag=1
  ParmsList=string.split(ParmsData[ParmsSetNum])
  for i in range(len(ParmsList)):
    x=ParmsList[i]
    Parms[i]=string.atof(x)
  PotInit(Parms)
  ParmsSetNum=ParmsSetNum+1
  return Parms

##################################################
# Initialize Parameters
# pass array of parameters (parms)
##################################################
#def PotInit(u0, alpha, r0, rc):
def PotInit(parms):
  global U0, Alpha, R0, PotInitFlag
  U0=parms[0]     # U0=u0
  Alpha=parms[1]  # Alpha=alpha
  R0=parms[2]     # R0=r0
#  RC=parms[3]     # RC=rc
  UC=U0*(1-exp(Alpha*(R0-RC)))*(1-exp(Alpha*(R0-RC)))
#  RCsq=RC*RC
  PotInitFlag=1

##################################################
# Morse Potential
##################################################
def Morse(Rij):
  global U0, Alpha, R0
#  if(not PotInitFlag):
#    print "Need to init pot before using"
#    return 0, array([0, 0, 0])
  Rsq=sum(Rij*Rij)
  if(Rsq>=RCsq):
    return 0, array([0.0, 0.0, 0.0])
  Fij=Rij.copy()
  R=sqrt(Rsq)
  Expr=exp(Alpha*(R0-R))
  OmExpr=1.0-Expr
  tmp=(2.0*Alpha*U0*Expr*OmExpr/R)
  Fij=Rij*(2*Alpha*U0*Expr*OmExpr/R)
  return U0*OmExpr*OmExpr-UC,Fij

##################################################
# Bulk Morse Potential
##################################################
def MorseBulk(Box, R):
  global U0, Alpha, R0, PotInitFlag,NumFC
#  if(not PotInitFlag):
#    print "Need to init pot before using"
#  if((R.nelements()%3)!=0):
#    print "Length of R needs to be a multiple of 3"
  U=0
  F=R.copy()*0.0
  Fij=array([0.0,0.0,0.0])
  NA=R.nelements()/3
  hBox=Box/2.0
  for ni in range(NA):
    for nj in range(ni):
      Rij=R[3*ni:3*ni+3]-R[3*nj:3*nj+3]
      for k in range(3):
        while Rij[k]<(-hBox[k]):
          Rij[k]+=Box[k]
        while Rij[k]>hBox[k]:
          Rij[k]-=Box[k]
#      if(not(abs(Rij[0])>RC or abs(Rij[1])>RC or abs(Rij[2])>RC)):
      if(not(Rij[0]>RC or  Rij[1]>RC or  Rij[2]>RC or \
             Rij[0]<-RC or Rij[1]<-RC or Rij[2]<-RC)):
        Rsq=sum(Rij*Rij)
        if(Rsq<RCsq):
          Rs=sqrt(Rsq)
          Expr=exp(Alpha*(R0-Rs))
          OmExpr=1.0-Expr
          Fij=Rij*(2*Alpha*U0*Expr*OmExpr/Rs)
          U+=U0*OmExpr*OmExpr-UC
          F[3*ni:3*ni+3]-=Fij[0:3]
          F[3*nj:3*nj+3]+=Fij[0:3]
  return U,F
