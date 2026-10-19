##################################################
# Point data class (selfosition, force, energy ...
##################################################
import numpy
from copy import copy
from Gman.func import vmag
##################################################
# Point class
##################################################
class point:
##################################################
  def Check(self):
    x="    "
    for k, v in self.__dict__.items():
         print(x, k)
         print(v)
##################################################
  def AddType(self,ntype,mass=1.0):
    try: self.MassType
    except: 
      self.MassType=[]
      for n in range (len(self.AtomType)):
          self.MassType.append(1.0)
    t=self.NumAtomTypes
    self.NumAtomTypes=t+1
    self.AtomType.append(ntype)
    self.MassType.append(mass)
    self.NumAtom.resize(len(self.NumAtom)+1)
    self.NumAtom[len(self.NumAtom)-1]=0
  ##################################################
  def RandReplace(self,num,fromtype,totype):
     for k in range(num):
        delllist=[]
        zumlist=[]
        dell=self.AtomType.index(fromtype)
        zum=self.AtomType.index(totype)
        for n in range(self.NumAtoms):
           if self.Type[n]==dell:
              delllist.append(n)
           if self.Type[n]==zum:
              zumlist.append(n)
        dn=delllist.__len__()
        i=int(dn*numpy.rand())
        self.Type[delllist[i]]=zum 
        self.NumAtom[dell]=self.NumAtom[dell]-1
        self.NumAtom[zum]=self.NumAtom[zum]+1
  ##################################################
  def Expand(self,numx,numy,numz):
    b=self.Box
    m=[numx,numy,numz]
    for n in range(3):
      if m[n] <= 1: 
        m[n]=1  
    self.Box=b*numpy.transpose([m,m,m])
    for n in range(3):
      if m[n] > 1: 
        L=len(self.R)
        ranc=m[n]*L
        self.Resize(ranc)
        for n1 in range(m[n]):
          for n2 in range(L):
            self.R[n2+n1*L]=self.R[n2]+n1*b[n]
    self.iBox=numpy.linalg.inv(self.Box)
  ##################################################
  def Resize(self,s):
    m=len(self.R)
    try: self.Mass
    except: self.Mass=numpy.ones(m) 
    try: self.MassD
    except: self.MassD=numpy.ones(numpy.shape(self.R)) 
    self.R=numpy.resize(self.R,(s,3))
    self.Type=numpy.resize(self.Type,(s,))
    self.Free=numpy.resize(self.Free,(s,))
    self.FreeD=numpy.resize(self.FreeD,(s,3))
    self.Mass=numpy.resize(self.Mass,(s,))
    self.MassD=numpy.resize(self.MassD,(s,3))
    self.NumAtom=(s*self.NumAtom)/m
    self.NumAtoms=(s*self.NumAtoms)/m
  ##################################################
  # note to self: sort is to sort self.Type, etc.. 
  def Sort(self):
    m=len(self.R)
    try: self.Mass
    except: self.Mass=numpy.ones(m) 
    try: self.MassD
    except: self.MassD=numpy.ones(numpy.shape(self.R)) 
    self.R=numpy.take(self.R,numpy.argsort(self.Type))
    self.Free=numpy.take(self.Free,numpy.argsort(self.Type))
    self.FreeD=numpy.take(self.FreeD,numpy.argsort(self.Type))
    self.Mass=numpy.take(self.Mass,numpy.argsort(self.Type))
    self.MassD=numpy.take(self.MassD,numpy.argsort(self.Type))
    self.Type=numpy.take(self.Type,nunpy.argsort(self.Type))
  ##################################################
  # note to self: sort is to sort self.AtomType, etc.. 
  def Resort(self):
    x=self.Type.__len__()
    t1=copy(self.AtomType)
    self.AtomType.sort() 
    t2=copy(self.AtomType)
    for n in range(x):
       x1=self.Type[n]
       self.Type[n]=t2.index(t1[x1])
  ###################################################
  def distanceTo(self, b):
    return vmag(b.R - self.R)
  ###################################################
  def distanceToSquared(self, b):
    return ((b.R - self.R)*(b.R - self.R)).sum()
  ###################################################
  def vectorTo(self, b):
    return b.R - self.R
  ###################################################
