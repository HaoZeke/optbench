##################################################
# Constrained molecular dynamics for H2O
##################################################
import math
import numpy
import Gman.func as Gfunc
##################################################
# Global variables for water molecules 
##################################################
#fdt=1e-5
#TolR=0.005
#TolRsq=TolR*TolR
#TolRP=1e-6
#ItrMax=1000
Theta=104.52*math.pi/180.0
rOH=0.9572
rOHsq=rOH*rOH
rCMO=2.0*math.cos(Theta/2.0)*rOH/18.0
rCH=math.sin(Theta/2.0)*rOH
rOHl=math.cos(Theta/2.0)*rOH
rH12=2.0*rCH
rH12sq=rH12*rH12
MH=1.0 
iMH=1.0/MH
MO=16.0
iMO=1.0/MO
##################################################
# Molecular Dynamics Step
##################################################
def mdstep(p,force,dt=0.1,tolR=1e-4,tolRP=1e-7,itrshakemax=1000):
  rattle1(p,dt,tolR,tolRP,itrshakemax)
  force(p)
  rattle2(p,dt,tolRP,itrshakemax)
##################################################
# Molecular dynamics run
##################################################
def md(p,force,dt=0.1,itrnum=100,ppd=100,vpd=0,thermV=0,bc=0):
  # Check to see if the force and velocity have been evaluated
  try: p.F
  except: force(p)
  try: p.V
  except: p.V=p.R.copy()*0.0
  print(p.V)
  for i in range(itrnum):
    # first check for a theromostat
    if(vpd!=0):
      if(i%vpd==0):
        try: p.V=thermV()
        except: print('No thermostat set')
    # velocity verlet algorithm
    mdstep(p,force,dt)
    # apply boundary conditions
    try: Gfunc.BC(p.R,p.Box,p.iBox)
    except: 0 # do nothing
##################################################
# Quick-Min Step
##################################################
def qmstep(p,force,dt=0.1,tolR=1e-4,tolRP=1e-7,itrshakemax=1000,quench='sys'):
  p.V=p.F*(p.V*p.F).sum()/(p.F*p.F).sum()
  # Zero any F antiparallel to the V
  # For the entire system
  if(quench=='sys'):
    if((p.V*p.F).sum()<0.0): p.V*=0.0
  # Molecule by molecule (not implemented)
  # Atom by atom
  elif(quench=='atm'):
    for i in range(p.NumAtoms):
      if((p.V[i]*p.F[i]).sum()<0.0): p.V[i]*=0.0
  # coordinate by coordinate
  elif(quench=='crd'):
    for i in range(p.NumAtoms):
      for j in range(3):
        if((p.V[i][j]*p.F[i][j]).sum()<0.0): p.V[i][j]*=0.0
  # do the regular mdstep
  rattle1(p,dt,tolR,tolRP,itrshakemax)
  force(p)
  rattle2(p,dt,tolRP,itrshakemax)
##################################################
# Rattle1: fix H2O bond lengths
##################################################
def rattle1(p,dt=0.1,tolR=0.005,tolRP=1e-6,itrshakemax=1000):
  global rOHsq,rH12sq,imH,imO,cfH1,cfH2,cfO
  idt=1.0/dt
  Q=p.R+p.V*dt+0.5*dt*dt*p.F/p.MassD
  NumMol=p.NumAtoms/3
  for i in range(NumMol):
    Done=0
    jO=2*NumMol+i
    jH1=2*i
    jH2=jH1+1
    while(Done==0):
      # Need to at ItrMax here
      Done=1
      Done*=rattlebond1(p.R[jH1],p.R[jO],p.F[jH1],p.F[jO],Q[jH1],Q[jO],iMH,iMO,rOHsq,idt,p.Box,p.iBox,tolR,tolRP)
      Done*=rattlebond1(p.R[jH2],p.R[jO],p.F[jH2],p.F[jO],Q[jH2],Q[jO],iMH,iMO,rOHsq,idt,p.Box,p.iBox,tolR,tolRP)
      Done*=rattlebond1(p.R[jH1],p.R[jH2],p.F[jH1],p.F[jH2],Q[jH1],Q[jH2],iMH,iMH,rH12sq,idt,p.Box,p.iBox,tolR,tolRP)
  A=p.F/p.MassD
  p.R+=p.V*dt+0.5*A*dt*dt
  p.V+=0.5*A*dt
##################################################
# Rattle1 for a particular molecule
##################################################
def rattlebond1(R1,R2,F1,F2,Q1,Q2,iM1,iM2,Rsq,idt,Box,iBox,tolR,tolRP):
  tolRsq=tolR*tolR
  R12=Gfunc.DBC(R1-R2,Box,iBox);
  Q12=Gfunc.DBC(Q1-Q2,Box,iBox);
  Dsq=Rsq-(Q12*Q12).sum()
  if(abs(Dsq)<Rsq*tolRsq):
    return 1
  Prj=(R12*Q12).sum()
  if(Prj<Rsq*tolRP):
    print('Contraint Failure in Rattle1')
  G12=Dsq/(2.0*(iM1+iM2)*Prj);
  d12=R12*G12
  Q1+=d12*iM1
  Q2-=d12*iM2
  F1+=d12*idt*idt*2.0
  F2-=d12*idt*idt*2.0
  return 0
##################################################
# Rattle2: fix H2O velocities
##################################################
def rattle2(p,dt=0.1,tolRP=1e-6,itrshakemax=1000):
  global rOHsq,rH12sq,imH,imO
  idt=1.0/dt
  A=p.F/p.MassD
  p.V+=A*dt*0.5
  NumMol=p.NumAtoms/3
  for i in range(NumMol):
    Done=0
    jO=2*NumMol+i
    jH1=2*i
    jH2=jH1+1
    while(Done==0):
      Done=1
      Done*=rattlebond2(p.R[jH1],p.R[jO],p.V[jH1],p.V[jO],p.F[jH1],p.F[jO],iMH,iMO,rOHsq,idt,p.Box,p.iBox,tolRP)
      Done*=rattlebond2(p.R[jH2],p.R[jO],p.V[jH2],p.V[jO],p.F[jH2],p.F[jO],iMH,iMO,rOHsq,idt,p.Box,p.iBox,tolRP)
      Done*=rattlebond2(p.R[jH1],p.R[jH2],p.V[jH1],p.V[jH2],p.F[jH1],p.F[jH2],iMH,iMH,rH12sq,idt,p.Box,p.iBox,tolRP)
##################################################
# Rattle2 for a particular molecule
##################################################
def rattlebond2(R1,R2,V1,V2,F1,F2,iM1,iM2,Rsq,idt,Box,iBox,tolRP):
  R12=Gfunc.DBC(R1-R2,Box,iBox)
  V12=Gfunc.DBC(V1-V2,Box,iBox)
  G12=-sum(R12*V12)/((iM1+iM2)*Rsq)
  if(abs(G12)<tolRP): return 1
  d12=R12*G12
  V1+=d12*iM1
  V2-=d12*iM2
  F1+=d12*idt*2.0
  F2-=d12*idt*2.0
  return 0

