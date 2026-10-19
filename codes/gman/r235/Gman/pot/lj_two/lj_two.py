##################################################
#  Lennard-Jones potential for two types of atoms
#
#  Version 1.1
#  by Dan. Last modified by Penghao 2010.05.16
##################################################
from tsse import *
from tsse.pot import pot
import numpy
import math
import sys
import lj_two_

##################################################
# Global Variables 
# Peng, X.Q. Composites: Part A 36 (2005) 859-874
##################################################
#EpsilonAA=1.14        # Epsilon: well depth
#EpsilonBB=1.0        # Epsilon: well depth
#EpsilonAB=0.223        # Epsilon: well depth
#SigmaAA=0.9          # Sigma: bond length parameter
#SigmaBB=0.865          # Sigma: bond length parameter
#SigmaAB=0.6          # Sigma: bond length parameter
#RC=10.0             # RC: Cutoff distance
##################################################
# Derived Global Variables
##################################################
#U0=4.0*Epsilon
#UC=U0*(math.pow(Sigma/RC,12)-math.pow(Sigma/RC,6))
#RCsq=RC*RC

##################################################
# Initialize Parameters
##################################################
class lj_two(pot):

	def __init__(self, epsilonAA=1.14, sigmaAA=0.9,epsilonBB=1.0, sigmaBB=0.865,epsilonAB=0.223, sigmaAB=0.6, rc=10.0):
#def init(epsilonAA=1.0, sigmaAA=1.0,epsilonBB=1.0, sigmaBB=1.0,epsilonAB=1.0, sigmaAB=1.0, rc=10.0):
	  #global RC,RCsq,EpsilonAA, SigmaAA, U0_AA, UC_AA, RCsq,EpsilonBB, SigmaBB, U0_BB, UC_BB,EpsilonAB, SigmaAB, U0_AB, UC_AB
	  pot.__init__(self)
          self.SigmaAA=sigmaAA
	  self.SigmaBB=sigmaBB
	  self.SigmaAB=sigmaAB
	  self.EpsilonAA=epsilonAA
	  self.EpsilonBB=epsilonBB
	  self.EpsilonAB=epsilonAB
	  self.RC=rc
	  self.U0_AA=4.0*self.EpsilonAA
	  self.UC_AA=self.U0_AA*(math.pow(self.SigmaAA/self.RC,12)-math.pow(self.SigmaAA/self.RC,6))
	  self.U0_BB=4.0*self.EpsilonAA
	  self.UC_BB=self.U0_BB*(math.pow(self.SigmaBB/self.RC,12)-math.pow(self.SigmaBB/self.RC,6))
	  self.U0_AB=4.0*self.EpsilonAB
	  self.UC_AB=self.U0_AB*(math.pow(self.SigmaAA/self.RC,12)-math.pow(self.SigmaAB/self.RC,6))
	  self.RCsq=self.RC*self.RC
	  self.PotInitFlag=1
###############################################
# Wrapper to fortran lennard-jones potential
###############################################
	def force(self,p):
	  """
          Performs a force call with the lennard-jones potential
	  Parameters:
              p:  tsse.point object
	  There are two types of atoms A and B with three kinds of potential: AA, BB and AB
	  The first NA atoms belong to type A, the rest N-NA belong to B
	  """
	  #print "Called!"
          #global RC,RCsq,EpsilonAA, SigmaAA, U0_AA, UC_AA, RCsq,EpsilonBB, SigmaBB, U0_BB, UC_BB,EpsilonAB, SigmaAB, U0_AB, UC_AB
	  El = p.unique("name")
          NA =0
	  for i in range(len(p)):
	         if p[i].name == El[0]: NA = NA+1

	  ncoo=len(p)*3
	  ra  =p.r.ravel()
	  fa  =ra*0.0
	  uRet=numpy.array([0],'d')
	  a1  =p.box[0]
	  a2  =p.box[1]
	  a3  =p.box[2]
	  norma1 = numpy.sqrt(sum(abs(a1)**2))
	  norma2 = numpy.sqrt(sum(abs(a2)**2))
	  norma3 = numpy.sqrt(sum(abs(a3)**2))
	  if self.RC > 0.5*min(norma1,norma2,norma3):
		  print('Error: cell size is smaller than twice cutoff radius')
		  sys.exit(1)
	  lj_two_.force(ra,fa,uRet,a1,a2,a3,self.U0_AA,self.SigmaAA,self.U0_BB,self.SigmaBB,self.U0_AB,self.SigmaAB,self.RC,NA)

	  p.f =numpy.resize(fa,(len(p),3))
#  print 'FORCE:  ',p.F
#  p.F*=p.FreeD  # Set frozen atoms to have no force
          try:
               p.f
          except:
               p.f = p.r * 0.0
	  p.u=uRet[0]
          self.accounting()

