##################################################
#  CAR potential
#
#  Version 1
#  Last modified by Albert 09.16.2007
##################################################
import numpy

##################################################
# Initialize Parameters
##################################################
#def init():
  # nothing
###############################################
# Calculate car potential
###############################################
def force(p):
  #Rflat=p.R.flat 
  ra=numpy.zeros(6,'d')
  fa=numpy.zeros(6,'d')
  uRet=numpy.array([0],'d')
  #x=Rflat[0]
  #y=Rflat[1]
  x=p.R[0][0]
  y=p.R[0][1]
  p.F=p.R*0.0
  p.F[0][0]=-0.24*x*(x*x+y*y)-y-18*math.exp(-y*y)*((x-3)*math.exp(-(x-3)*(x-3))+(x+3)*math.exp(-(x+3)*(x+3)))
  p.F[0][1]=-0.24*y*(x*x+y*y)-x-18*y*math.exp(-y*y)*(math.exp(-(x-3)*(x-3))+math.exp(-(x+3)*(x+3)))
#  try: p.F*=p.FreeD
#  except: p.FreeD=numpy.ones((p.NumAtoms,p.dim),'d')
  try:
    p.F*=p.FreeD  # Set frozen atoms to have no force
  except:
    pass
  p.U=0.06*(x*x+y*y)*(x*x+y*y)+x*y-9*math.exp(-y*y)*(math.exp(-(x-3)*(x-3))+math.exp(-(x+3)*(x+3)))

