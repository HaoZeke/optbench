##################################################
# Functions to read and write con files
##################################################
import numpy
##################################################
# Function to load a con file
##################################################
def loadcon(p,filename,file=0):
  p.PointType='CON'
  # if an open file is passed in, use that, otherwise open filename
  if(file==0):
    try:
      con=open(filename,'r')
    except:
      print("Could not open file ",filename)
      return
  else:
    con=file
  # read the con file data
  p.ConHeader1=con.readline()[:-1]
  p.ConHeader2=con.readline()[:-1]
  # Determine how many dimentions 
  tmp=numpy.array(con.readline().split())
  for i in range(len(tmp)):
    p.dim=i+1
    try: float(tmp[i])
    except:
      p.dim=i
      break 
  BoxLength=numpy.zeros(p.dim)
  for i in range(p.dim):
    BoxLength[i]=float(tmp[i])
#  BoxLength=numpy.array([ float(f) for f in con.readline().split()[0:3] ])
  BoxAngle=numpy.array([ float(f) for f in con.readline().split()[0:p.dim] ])
  for i in range(p.dim):
    if(BoxAngle[i]!=90):
      print("Non orthogonal box in con files not yet implemented.")
  p.Box=numpy.zeros((p.dim,p.dim),'d')
  for i in range(p.dim):
    p.Box[i][i]=BoxLength[i]
  p.iBox=numpy.linalg.inv(p.Box)
  p.ConHeader3=con.readline()[:-1]
  p.ConHeader4=con.readline()[:-1]
  p.NumAtomTypes=int(con.readline().split()[0])
  p.NumAtom=numpy.array([ int(f) for f in con.readline().split()[0:p.NumAtomTypes] ])
  p.NumAtoms=p.NumAtom.sum()
  p.Type=numpy.zeros((p.NumAtoms),'i')
  p.R=numpy.zeros((p.NumAtoms,p.dim),'d')
  p.Mass=numpy.zeros((p.NumAtoms),'d')
  p.MassD=numpy.zeros((p.NumAtoms,p.dim),'d')
  p.Free=numpy.zeros((p.NumAtoms),'d') 
  p.FreeD=numpy.zeros((p.NumAtoms,p.dim),'d')
  if(p.NumAtomTypes!=len(p.NumAtom)):
    print("Warning: Number of atom types listed does not match number found in con file.")
  p.MassType=numpy.array([ float(f) for f in con.readline().split()[0:p.NumAtomTypes] ])
  if(p.NumAtomTypes!=len(p.MassType)):
    print("Warning: Number of atom masses listed does not match number found in con file.")
  p.AtomType=[]
  p.AtomTypeConHeader=[]
  AtomNum=-1
  for i in range(p.NumAtomTypes):
    p.AtomType.append(con.readline().split()[0])
    p.AtomTypeConHeader.append(con.readline()[:-1])
    for j in range(p.NumAtom[i]):
      AtomNum+=1
      p.Type[AtomNum]=i
      line=numpy.array([ float(f) for f in con.readline().split() ])
      p.R[AtomNum]=line[0:p.dim]
      p.Mass[AtomNum]=p.MassType[i]
      p.MassD[AtomNum]=[p.MassType[i]]*p.dim
      if(line[p.dim]==1): p.Free[AtomNum]=0
      else: p.Free[AtomNum]=1
      for k in range(p.dim):
        p.FreeD[AtomNum][k]=p.Free[AtomNum]
##################################################
# Function to save a con file
##################################################
def savecon(p,filename,w='w'):
  con=open(filename,w)
  try: print(p.ConHeader1, file=con)
  except: print("10000 RANDOM NUMBER SEED", file=con)
  try: print(p.ConHeader2, file=con)
  except: print("0.0000 TIME", file=con)
  try: p.dim
  except: p.dim=len(p.R[0])
  # Note:  This will only for for orthogonal boxes 
  BoxDiag=numpy.zeros((p.dim),'d')
  for i in range(p.dim): BoxDiag[i]=p.Box[i][i]
  print(" ".join(['%20.14f' % s for s in BoxDiag]), file=con)
  Angle=numpy.zeros((p.dim),'d')+90.0
  print(" ".join(['%20.14f' % s for s in Angle]), file=con)
  try: print(p.ConHeader3, file=con)
  except: print("0 0", file=con)
  try: print(p.ConHeader4, file=con)
  except: print("0 0 0", file=con)
  print(p.NumAtomTypes, file=con)
  print(" ".join(['%s' % s for s in p.NumAtom]), file=con)
  try: print(" ".join(['%s' % s for s in p.MassType]), file=con)
  except: print(" ".join(['1.0']*p.NumAtomTypes), file=con)
  NumAtom=-1
  for i in range(p.NumAtomTypes):
    print(p.AtomType[i], file=con)
    try: print(p.AtomTypeConHeader[i], file=con)
    except: print("Coordinates of Component ",i+1, file=con)
    for j in range(p.NumAtom[i]):
      NumAtom+=1
      line=" ".join(['%20.14f' % s for s in p.R[NumAtom]])+"  "
      line+="  ".join(['%d' % s for s in [-1*(p.Free[NumAtom]-1),NumAtom]])
      print(line, file=con)
