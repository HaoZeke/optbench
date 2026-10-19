#############################################
# Voter eam potential
#############################################
import numpy
import string
import sys
import eamvoter_

ntype=0
maxtyp=0
iwrt=1
ietype=0
rcut=0
amu=0
azero=0
Mass=1.0*numpy.array(0)
A0=1.0*numpy.array(0)
##############################################
#def checkvalve():
#     global Mass,A0
#     print Mass
#############################################
# PotSet: sets the potential types
#############################################
def potset(El):
#   print sys.path
   portapotdir=sys.modules['Gman.pot'].__path__[0]+'/eamvoter/portapot/'
#   print "loading EAM potential from:\n",portapotdir
   global ntype,maxtyp,iwrt,ietype,rcut,amu,azero
   temp=numpy.zeros(len(El),'d')
   Mbutt=temp.copy()
   Anull=temp.copy()

   ntype=len(El)
   maxtyp=len(El)
   iwrt=1
   ietype=0
#   rcut=numpy.zeros([maxtyp,maxtyp],'d')
   rcut=numpy.zeros([maxtyp*maxtyp],'d')
   amu=numpy.ones(maxtyp,'d')
   azero=numpy.ones(maxtyp,'d')

   targ=El
   for i in range(len(El)):
     q=80-len(El[i])
     targ[i]=string.lower(El[i])+q*" "
   q=256-len(portapotdir)
   potdir=portapotdir+q*" "
   potnam=numpy.array(targ,'c')
#   eamvoter_.eam_setup(Mbutt,Anull,potnam)
#   eamvoter_.eam_setup(Mbutt,Anull,potnam,potdir)

#   print "amu: ",amu
#   print "rcut: ",rcut
#   print "potnam ",potnam
#   print "potdir: ",potdir
#   print 'before eam_setup'
#   eamvoter_.eam_setup(ntype,maxtyp,potnam,potdir,iwrt,ietype,rcut,amu,azero)
   eamvoter_.eam_setup(amu,azero,rcut,potnam,potdir)
#   eamvoter_.eam_setup(amu,azero,potnam,potdir)
#   print 'after eam_setup'
#   print "amu: ",amu
#   print "azero: ",azero
#   print "rcut: ",rcut

#############################################
# set: sets the potential type from AtomType
#############################################
def init(p):
  potset(p.AtomType)
#############################################
#def force(R, B, E):
#############################################
def force(p):
     global ntype,maxtyp,iwrt,ietype,rcut,amu,azero
#     print "into force (python)"
     natom=p.NumAtoms
     hyperratio=1.0
     uret=numpy.array([0],'d')
     [X,Y,Z]=numpy.transpose(p.R)
     itype=X.copy()
     FX=X.copy()
     FY=X.copy()
     FZ=X.copy()
     for i in range(len(X)):
       itype[i]=1+p.Type[i]
     tax=p.Box[0][0]
     tay=p.Box[1][1]
     taz=p.Box[2][2]
#     p.U=eamvoter_.force(X,Y,Z,I,tax,tay,taz,FX,FY,FZ)
#     print "X ",X
#     print "Y ",Y
#     print "Z ",Z
#     print "itype ",itype
#     print "tax ",tax
#     print "tay ",tay
#     print "taz ",taz
#     print "ietype ",ietype
#     print "maxtyp ",maxtyp
#     print "rcut ",rcut
#     print "uret ",uret
#     print "before g0 (python)"
#     eamvoter_.g0(X,Y,Z,itype,tax,tay,taz,ietype,maxtyp,rcut,uret,FX,FY,FZ,hyperratio)
     eamvoter_.g0(X,Y,Z,itype,tax,tay,taz,ietype,uret,FX,FY,FZ,hyperratio)
#     print "after g0 (python)"
     # Check this force array
     p.F=-numpy.transpose((FX,FY,FZ))
#     p.F=-numpy.reshape(grad,(p.NumAtoms,p.dim))
     p.U=uret[0]
     # Convert from Hartree -> eV
     p.U*=27.21
     p.F*=27.21
     try: p.F*=p.FreeD
     except: p.FreeD=numpy.ones((p.NumAtoms,p.dim),'d')
#############################################
