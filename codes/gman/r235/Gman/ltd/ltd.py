##################################################
# Long time dynamics for vasp
##################################################
import string
##################################################
# Global Variables
##################################################
Seed=0
Time=0
Pref=1e12
dR=0.1
dU=0.02
RandR=3
MaxPr=10  # get this from Gincar
MaxSt=100
NewSt=CurSt=NumSt=0
DepositFlag=0
##################################################
# Allocate Memory
##################################################
St=data.point()[MaxSt]
##################################################
# Initializations
##################################################
random.seed(Seed)
##################################################
# Set Parameters
##################################################
def SetParm(Parm,Val):
  global Time, Pref
  if Parm=='Time':
    Time=string.atof(Val)
  elif Parm=='Pref':
    Pref=string.atof(Val)
##################################################
# Main function
##################################################
def Run():
  # do a full minimization on the initial file
  print('Running')
