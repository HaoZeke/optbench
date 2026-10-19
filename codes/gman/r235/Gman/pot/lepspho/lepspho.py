##################################################
#  LEPS+HO potential
#
#  Version 1
#  Last modified by Graeme 04.23.2007
##################################################
import numpy
import lepspho_

##################################################
# Initialize Parameters
##################################################
#def init():
  # nothing
###############################################
# Wrapper to fortran LEPS+HO potential
###############################################
def force(p):
  Rflat=p.R.flat
  if(p.R.size>3):
    print("Length of p.R needs to be 2 or 3 for the LEPS+HO potential")
  ra=numpy.zeros(6,'d')
  fa=numpy.zeros(6,'d')
  uRet=numpy.array([0],'d')
  ra[0]=Rflat[0]
  ra[3]=Rflat[1]
  lepspho_.force(ra,fa,uRet)
  p.F=p.R*0.0
  p.F[0][0]=fa[0]
  p.F[0][1]=fa[3]
#  try: p.F*=p.FreeD
#  except: p.FreeD=numpy.ones((p.NumAtoms,p.dim),'d')
  try:
    p.F*=p.FreeD  # Set frozen atoms to have no force
  except:
    pass
  p.U=uRet[0]

