##################################################
#  Ewald-Buckingham potential
#  wrapping the tad code obtained from Blas
#  Version 1
#  Last modified by GH on Jan 13, 2005
##################################################
import numpy
import buck_
##################################################
## Global Variables shared by ewald_buck potential
## in its intialization and force subroutines
##################################################
rcut=numpy.array([])  
amu=numpy.array([])   #amu(maxtyp)
taxes=numpy.array([])  # never used
azero=numpy.array([]) #azero(maxtyp)
PotInitFlag=0
maxtyp=0
natom=0
x=numpy.array([])
y=numpy.array([])
z=numpy.array([])
grad=numpy.array([])
Itype=numpy.array([])
##################################################
## Initialize the potentials
## pass system basic information into the fortran
## initializer
##################################################
#def init(ntype, natom, typeatom):
def init(p):
  global rcut,amu,azero,taxes,maxtyp,natom,x,y,z,grad,Itype
  natom=p.NumAtoms
  maxtyp=p.NumAtomTypes   #exactly the number of atom types
  Itype=p.Type  #Itype(i) is the type of atom i
  rcut=numpy.zeros(maxtyp, 'd') #rcut(maxtyp, maxtyp)
  for item in rcut:
      item=numpy.zeros(maxtyp, 'd')
  amu=numpy.zeros(maxtyp, 'd')
  azero=numpy.zeros(maxtyp, 'd')
  taxes=numpy.zeros(3,'d') #never used
  buck_.prmst72(ntype,maxtyp,rcut,amu,azero,taxes)
  #Note: taxes would bring in 3 values if used
  x=numpy.zeros(natom, 'd')
  y=numpy.zeros(natom, 'd')
  z=numpy.zeros(natom, 'd')
  grad=numpy.zeros(3,'d')
  for i in grad:
      i=numpy.zeros(natom, 'd')    #grad(3,natom)
  PotInitFlag=1
###############################################
# force call
###############################################
#def force(Box, R):
def force(p):
  global rcut,amu,azero,taxes,maxtyp,natom,x,y,z,grad,Itype
# if(not PotInitFlag):
# print "Need to init pot before using"  
#  if((len(R)%3)!=0):
#      print "Length of R needs to be a multiple of 3"
  R=p.R.flat
  if(natom!=(len(R)/3)):
      print("check the size of R array, sth is wrong.")
  nmove=natom
  F=R.copy()*0.0 # R and Box must be passed in as Numeric arrays
  U_dummy=numpy.array([0.0])
  for i in range(natom):
      x[i]=R[3*i-2]
      y[i]=R[3*i-1]
      z[i]=R[3*i]
  buck_.g72(natom,nmove,x,y,z,Itype,maxtyp,Box[1],Box[2],Box[3],rcut,U_dummy,grad)
  for i in range(natom):
      F[3*i-2]=-1.0*grad(1,i)
      F[3*i-1]=-1.0*grad(2,i)
      F[3*i]=-1.0*grad(3,i)
  U=U_dummy[0]
# print 'U=', U
  return U,F

