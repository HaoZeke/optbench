##################################################
# Functions to read and write vasp files
##################################################
import numpy
import string
##################################################
# Function to load a poscar file
##################################################
def loadvasp(p,filename,file=0):
  p.PointType='VASP'
  if(file==0):
    poscar=open(filename,'r')
  else:
    poscar=file
  p.VaspHeader=poscar.readline()[:-1]
  p.AtomType=string.split(p.VaspHeader)
  p.NumAtomTypes=len(p.AtomType)
  scale=string.atof(poscar.readline())
  p.Box=numpy.zeros((3,3),'d')
  for i in range(3):
    p.Box[i]=numpy.array([ float(f) for f in poscar.readline().split() ])*scale
  p.Box=numpy.transpose(p.Box)
  p.iBox=numpy.linalg.inv(p.Box)
  p.NumAtom=numpy.array([ int(f) for f in poscar.readline().split() ])
  p.NumAtoms=p.NumAtom.sum()
  sel=poscar.readline()[0]
  p.SelectiveFlag=(sel=='s' or sel=='S')
  if(not p.SelectiveFlag): car=sel
  else: car=poscar.readline()[0]
  p.DirectFlag=not(car=='c' or car=='C' or car=='k' or car=='K')
  p.R=numpy.zeros((p.NumAtoms,3),'d')
  p.Type=numpy.zeros((p.NumAtoms))
  p.Free=numpy.ones((p.NumAtoms))
  p.FreeD=numpy.ones((p.NumAtoms,3))
  type=-1
  tmp=0
  for i in range(p.NumAtoms):
    if(i>=tmp):
      type+=1
      tmp+=p.NumAtom[type]
    p.Type[i]=type
    try: line=poscar.readline().split()
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
    else: p.R*=scale
##################################################
# Function to save a poscar file
##################################################
def savevasp(p,filename,w='w'):
  p.Sort()
  poscar=open(filename,w)
  try: p.DirectFlag
  except: p.DirectFlag=False
  try: p.VaspHeader
  except: p.VaspHeader=" ".join(p.AtomType) 
  try: p.SelectiveFlag
  except: p.SelectiveFlag=True
  print(p.VaspHeader, file=poscar)
  print(1.0, file=poscar)
  for i in range(3):
    print(" ".join(['%20.14f' % s for s in p.Box[i]]), file=poscar)
  print(" ".join(['%s' % s for s in p.NumAtom]), file=poscar)
  if(p.SelectiveFlag):
    print('Selective Dynamics', file=poscar)
  if(p.DirectFlag):
    print('Direct', file=poscar)
    for i in range(p.NumAtoms):
      pos=numpy.dot(p.iBox,p.R[i])
      posline=" ".join(['%20.14f' % s for s in pos])+" "
      if(p.SelectiveFlag):
        for j in range(3):
          if(p.FreeD[i][j]): posline+='   T'
          else: posline+='   F'
      print(posline, file=poscar)
  else:
    print('Cartesian', file=poscar)
    for i in range(p.NumAtoms):
      posline=" ".join(['%20.14f' % s for s in p.R[i]])+" "
      if(p.SelectiveFlag):
        for j in range(3):
          if(p.FreeD[i][j]): posline+='   T'
          else: posline+='   F'
      print(posline, file=poscar)
