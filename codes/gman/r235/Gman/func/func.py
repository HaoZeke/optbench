#################################################
# General Gman functions
##################################################
import numpy
from math import sqrt
##################################################
# Periodic boundary conditions [0..1]
##################################################
def BBC(v,box,ibox=0):
  if(ibox is 0):
    ibox=numpy.linalg.inv(box)
  vdir=numpy.dot(v,ibox)
  vdir=(vdir%1.0+1.0)%1.0
  return numpy.dot(vdir,box)
##################################################
# Periodic boundary conditions [-0.5..0.5]
##################################################
def DBC(v,box,ibox=0):
  if(ibox is 0):
    ibox=numpy.linalg.inv(box)
  vdir=numpy.dot(v,ibox)
  vdir=(vdir%1.0+1.5)%1.0-0.5
  return numpy.dot(vdir,box)
##################################################
# Expand the cell along the lattice vectors
##################################################
def expand(p,numx,numy,numz):
  b=p.Box
  m=[numx,numy,numz]
  for n in range(3):
    if m[n] <= 1:
      m[n]=1
  p.Box=b*numpy.transpose([m,m,m])
  r1=p.R.copy()
  for n in range(3):
    if m[n] > 1:
      L=len(r1)
      ranc=m[n]*L
      r2=numpy.resize(r1,(ranc,3))
      for n1 in range(m[n]):
        for n2 in range(L):
          nt=r1[n2]
          ns=nt+b[n]
          r2[L+n2]=ns
      r1=r2.copy()
      print(r1)
  p.R=r1.copy()
##################################################
# Vector projection
##################################################
def vproj(v1,v2):
  return v2*((v1*v2).sum()/((v2*v2).sum()))
##################################################
# Unit vector
##################################################
def vunit(v):
  return v/sqrt((v*v).sum())
##################################################
# Vector magnitude
##################################################
def vmag(v):
  return sqrt((v*v).sum())
##################################################
# Vector magnitude squared
##################################################
def vrsq(v):
  return (v*v).sum()
##################################################
# Vector dot product
##################################################
def vdot(v1,v2):
  return (v1*v2).sum()
##################################################
# Random vector
##################################################
def vrand(v):
  vtemp=numpy.random.randn(v.size)
  return vtemp.reshape(v.shape)
##################################################
# flatten a 2D matrix
##################################################
def flatten(v):
  dim=numpy.shape(v)
  d=numpy.zeros(dim[0]*dim[1])
  kk=0
  for n1 in range(dim[0]):
    for n2 in range(dim[1]):
      d[kk]=v[n1,n2]
      kk+=1
  return d
#  d=numpy.zeros(numpy.shape(v))
#  d=flatten(v)
#  for n1 in range(size/3):
#    for n2 in range(3):
#      for n3 in range(size):
#        for n4 in range(size):
#          d[n1,n2]+=A[n1,n4]*v[n5,n6]
#  print d
