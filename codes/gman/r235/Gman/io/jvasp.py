##################################################
# Functions to read and write jvasp files
##################################################
import numpy
import string
##################################################
# Function to load a Jvasp file
################################################## 
def loadjvasp(p, filename,file=0):
  p.PointType='Jvasp'
  if(file==0):
    jvasp=open(filename,'r')
  else:
    jvasp=file
  Header=jvasp.readline().split()
  p.JvaspHeader=Header[0]
  AtomTypeNumText=Header[1].split('=')
  AtomTypeText=Header[2].split('=')
  if(AtomTypeNumText[0] != 'NCLASS' or AtomTypeText[0] != 'ATOM'):
    print('Error: Jvasp header missing NCLASS or ATOM tag')
  p.NumAtomTypes=string.atoi(AtomTypeNumText[1])
  p.AtomType=[]
  p.AtomType.append(AtomTypeText[1])
  p.AtomType+=Header[3:]
  p.NumAtom=numpy.array([int(i) for i in jvasp.readline().split()])
  p.NumAtoms=p.NumAtom.sum()
  if(len(p.NumAtom) != len(p.AtomType) or len(p.NumAtom) != len(p.NumAtom)):
    print('Error: Number of atom types found does not match specified number')
  sel=jvasp.readline()[0]
  p.SelectiveFlag=(sel=='s' or sel=='S')
  if(not p.SelectiveFlag): car=sel
  else: car=poscar.readline()[0]
  p.DirectFlag=not(car=='c' or car=='C' or car=='k' or car=='K')
  BlankLine=jvasp.readline()
  scale=string.atof(jvasp.readline())
  p.Box=numpy.zeros((3,3),'d')
  for i in range(3):
    p.Box[i]=numpy.array([float(f) for f in jvasp.readline().split()])*scale
  p.Box=numpy.transpose(p.Box)
  p.R=numpy.zeros((p.NumAtoms,3),type=float)
  p.Type=numpy.zeros((p.NumAtoms))
  p.Free=numpy.ones((p.NumAtoms))
  p.FreeD=numpy.ones((p.NumAtoms,3))
  BlankLine=jvasp.readline()
  type=-1
  tmp=0
  for i in range(p.NumAtoms):
    if(i>=tmp):
      type+=1
      tmp+=p.NumAtom[type]
    p.Type[i]=type
    try: line=jvasp.readline().split()
    except: print('Error: More atoms (',p.NumAtoms,') than positions (',i,').')
    if(p.SelectiveFlag): assert len(line)>=6
    else: assert len(line)>=3
    pos=line[0:3]
    if(p.SelectiveFlag):
      sel=line[3:7]
      for j in range(3):
        if(sel[j]=='T' or sel[j]=='t'): p.FreeD[i][j]=1
        elif(sel[j]=='F' or sel[j]=='f'): p.FreeD[i][j]=0
        else: print('Error: Constraint flag in Jvasp file is not [T,t,F,f].')
      p.Free[i]*=p.FreeD[i][j]
    p.R[i]=numpy.array([float(f) for f in pos])
    if(p.DirectFlag): p.R[i]=numpy.dot(p.Box,p.R[i])
    else: Self.R*=scale
##################################################
# Function to save a Jvasp file
##################################################
def savejvasp(p,filename,w='w'):
  try: p.iBox
  except: p.iBox=numpy.linalg.inv(p.Box)
  jvasp=open(filename,w)
  try: Header=p.JvaspHeader
  except: Header="Generated_By_Gman"
  print(Header+("  NCLASS=%d" % p.NumAtomTypes)+"  ATOM="+" ".join(['%s' % s for s in p.AtomType]), file=jvasp)
  print(" ".join(['%s' % s for s in p.NumAtom]), file=jvasp)
# It's not even clear if Jvasp can have selective dynamics (or if it belongs in this line)
#  try: SelFlag=p.SelectiveFlag
#  except: SelFlag=0
#  if(SelFlag):
#    print >> jvasp, 'Selective Dynamics'
#  Need to add support for Cartesian if appropriate here
#    print >> poscar, 'Cartesian'
  print('Direct', file=jvasp)
  print('', file=jvasp)
  print(1.0, file=jvasp)
  for i in range(3):
    print(" ".join(['%20.14f' % s for s in p.Box[i]]), file=jvasp)
  print('', file=jvasp)
  for i in range(p.NumAtoms):
    pos=numpy.dot(p.iBox,p.R[i])
    posline=" ".join(['%20.14f' % s for s in pos])+" "
#    if(SelFlag):
#      for j in range(3):
#        if(p.FreeD[i][j]): posline+='   T'
#        else: posline+='   F'
    print(posline, file=jvasp)
