#############################################
# BJX water potential
#############################################
import numpy
import tip4p_

#############################################
# PotSet: sets the potential types
#############################################
def init():
  tip4p_.potinit()
#############################################
def force(p):
  ncoo=p.NumAtoms*3
  Rflat=p.R.flat
  raOri=numpy.zeros(ncoo, numpy.float)
  fa=numpy.zeros(ncoo, numpy.float)
  for i in range(ncoo):
    raOri[i]=Rflat[i]
  uRet=numpy.array([0.0])
  tax=p.Box[0][0]
  tay=p.Box[1][1]
  taz=p.Box[2][2]
  tip4p_.gagafe(raOri,fa,uRet,tax,tay,taz)
  p.F=numpy.resize(fa,(p.NumAtoms,3))
  p.U=uRet[0]
#############################################
