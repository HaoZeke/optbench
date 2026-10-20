##################################################
#  no potential
#
#  Version 1
#  Last modified by Dan 06.15.2007
##################################################
import numpy
##################################################
# Initialize Parameters
##################################################
# Wrapper to fortran Mueller-Brown potential
###############################################
def force(p):
  p.F=p.R*0.0
  p.F*=p.FreeD  # Set frozen atoms to have no force
  p.U=0.0

