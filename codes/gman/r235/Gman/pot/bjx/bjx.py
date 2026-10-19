#############################################
# BJX water potential
#############################################
import numpy
import bjx_
#############################################
# PotSet: sets the potential types
#############################################
def init():
  bjx_.potinit()
#############################################
def force(p):
  ncoo=p.NumAtoms*3
  Rflat=p.R.flat
  raOri=numpy.zeros(ncoo, numpy.float)
  fa=numpy.zeros(ncoo, numpy.float)
  for i in range(ncoo):
    raOri[i]=Rflat[i]
  uRet=numpy.array([0.0])
  ax=p.Box[0][0]
  ay=p.Box[1][1]
  az=p.Box[2][2]
  bjx_.gagafe(raOri,fa,uRet,ax,ay,az)
  p.F=numpy.array(fa)
  p.F=numpy.resize(fa,(p.NumAtoms,3))
  p.U=uRet[0]
#############################################
