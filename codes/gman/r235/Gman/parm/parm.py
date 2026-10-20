#############################################
#  Gman
#    Module: reads paramters from GINCAR 
#
#  Graeme Henkelman
#  Lijun Xu
#  William Stier
#
# Last modified by WS 2004.07.08
#############################################
import string
parameter={}
def LoadParameters(infile='GINCAR'):
    input = open(infile, 'r')
    parameterfile = input.readlines()
    input.close
    for line in parameterfile: 
      delima = string.find(line,"=")
      delimb = string.find(line,"#")
      key=line[:delima]
      value=line[delima+1:delimb]
      parameter[line[:delima]] = line[delima+1:delimb]
#############################################

