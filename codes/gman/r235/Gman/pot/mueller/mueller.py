##################################################
#  Mueller-Brown potential
#
#  Version 1
#  Last modified by Dan 05.11.2007
##################################################
import numpy
import mueller_

##################################################
# Initialize Parameters
##################################################
#def init():
  # nothing
###############################################
# Wrapper to fortran Mueller-Brown potential
###############################################
def force(p):
#  print 'shape', numpy.shape(p.R)
  Rflat=p.R.flat
  if(p.R.size>3):
    print("Length of p.R needs to be 2 or 3 for the Mueller-Brown potential")
  ra=numpy.zeros(3,'d')
  fa=numpy.zeros(3,'d')
  uRet=numpy.array([0],'d')
  ra[0]=Rflat[0]
  ra[1]=Rflat[1]
  mueller_.force(ra,fa,uRet)
  p.F=p.R*0.0
  p.F[0][0]=fa[0]
  p.F[0][1]=fa[1]
  try: p.FreeD  # Set frozen atoms to have no force
  except: p.FreeD = numpy.ones(numpy.shape(p.R),'d')
  conv = 0.010364 # KJ/mol to eV
  p.U=uRet[0]*conv
  p.F*=conv
  try: p.F*=p.FreeD
  except: p.FreeD=numpy.ones((p.NumAtoms,p.dim),'d')

