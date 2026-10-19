#############################################
# EAM Al potential
#############################################
import numpy
import al_
#############################################
# PotSet: sets the potential types
#############################################
def init():
  al_.potinit()
#############################################
def force(p):
  ncoo=p.NumAtoms*3
  Rflat=p.R.flat
  ra=numpy.zeros(ncoo, 'd')
  fa=numpy.zeros(ncoo, 'd')
  for i in range(ncoo):
    ra[i]=Rflat[i]
  uRet=numpy.array([0.0])
  ax=p.Box[0][0]
  ay=p.Box[1][1]
  az=p.Box[2][2]
  al_.force(ra,fa,uRet,ax,ay,az)
  p.F=numpy.resize(fa,(p.NumAtoms,3))
  p.F*=p.FreeD  # Set frozen atoms to have no force
  try: p.F*=p.FreeD
  except: p.FreeD=numpy.ones((p.NumAtoms,p.dim),'d')
  p.U=uRet[0]
#############################################
