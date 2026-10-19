##################################################
# Some extra math functions used in Gman
##################################################
from . import math
#import numarray as na
import numpy
from random import random
from random import normalvariate
##################################################
# sign: returns 1 or -1 based on the sign of a
##################################################
def sign(a):
  if(a<0.0): return -1.0
  return 1.0
##################################################
# sign: returns a with the sign of b
##################################################
#def sign(a,b):
#  if(b<0.0): return -fabs(a)
#  return fabs(a)
##################################################
# Gaussian distributed random number
##################################################
def gaussn(): return normalvariate(0.0,1.0)
##################################################
# RandFlux return a flux weighted number
# should use random.expovariate()
##################################################
def randflux(): return math.sqrt(-2.0*log(1.0-random()))
##################################################
# Gaussian distributed vector (numpy)
##################################################
def gaussv(dim): 
  return numpy.array([normalvariate(0.0,1.0) for i in range(dim)])
##################################################
# Periodic boundary conditions [-1/2..1/2]
##################################################
def DBC(v,box,ibox):
  v1=0
# adding transformation here.
##################################################
# Periodic boundary conditions [0..1]
##################################################
def BBC(v,box,ibox): 
  v1=0

