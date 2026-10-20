#############################################
# Voter eam potential
#############################################
import numpy
import string
import sys
import eamvoter_

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
   global Mass,A0
   temp=numpy.zeros(len(El),'d')
   Mbutt=temp.copy()
   Anull=temp.copy()
   targ=El
   for i in range(len(El)):
     q=80-len(El[i])
     targ[i]=string.lower(El[i])+q*" "
   q=256-len(portapotdir)
   potdir=portapotdir+q*" "
   potnam=numpy.array(targ,'c')
#   eamvoter_.eam_setup(Mbutt,Anull,potnam)
   eamvoter_.eam_setup(Mbutt,Anull,potnam,potdir)
   Mass=Mbutt
   A0=Anull
#############################################
# set: sets the potential type from AtomType
#############################################
def init(p):
  potset(p.AtomType)
#############################################
#def force(R, B, E):
#############################################
def force(p):
     [X,Y,Z]=numpy.transpose(p.R)
     I=X.copy()
     FX=X.copy()
     FY=X.copy()
     FZ=X.copy()
     for i in range(len(X)):
       I[i]=1+p.Type[i]
     tax=p.Box[0][0]
     tay=p.Box[1][1]
     taz=p.Box[2][2]
     p.U=eamvoter_.force(X,Y,Z,I,tax,tay,taz,FX,FY,FZ)
     # Check this force array
     p.F=numpy.transpose((FX,FY,FZ))
     # Convert from Hartree -> eV
     p.U*=27.21
     p.F*=27.21
     try: p.F*=p.FreeD
     except: p.FreeD=numpy.ones((p.NumAtoms,p.dim),'d')
#############################################
