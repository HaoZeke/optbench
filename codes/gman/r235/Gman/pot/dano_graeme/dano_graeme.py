##################################################
#  dano_graeme potential
#
#  Version 1
# Created by dano: 8-28-2010
##################################################
import numpy
import dano_graeme_

##################################################
# Initialize Parameters
##################################################
def init():
  print("dano_graeme potential")
  # nothing
###############################################
# Wrapper to fortran
###############################################
def force(p):
  if(p.R.size>3):
    print("Length of p.R need to be two dim")
  ra=numpy.zeros(3,'d')
  fa=numpy.zeros(3,'d')
  ra[0]=p.R[0][0]
  ra[1]=p.R[0][1]
  uRet=numpy.array([0],'d')
  dano_graeme_.force(ra,fa,uRet)
  p.F=p.R*0.0
  p.F[0][0]=fa[0]
  p.F[0][1]=fa[1]
#  try: p.F*=p.FreeD
#  except: p.FreeD=numpy.ones((p.NumAtoms,p.dim),'d')
  try:
    p.F*=p.FreeD  # Set frozen atoms to have no force
  except:
    pass
  p.U=uRet[0]

