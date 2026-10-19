#import sys

callBack = None
forceCount = 0


def force(p, count=True):
  forceCall = globals()["callBack"]
  forceCall(p)
  if count:
    globals()["forceCount"] += 1

def countReset():
  globals()["forceCount"] = 0

def count():
  return globals()["forceCount"]


#############################################
# Potential set routine
#############################################
def set(potname,p=0):
  # Hannes' eam Al potential
  if(potname=='al'):
    try:
      from . import al
      al.init()
      globals()["callBack"] = al.force
      return force
    except: 
      print('Failed to load potential: al')
  # BJX water potential
  elif(potname=='bjx'):
    try: 
      from . import bjx
      bjx.init()
      globals()["callBack"] = bjx.force
      return force
    except:
      print('Failed to load potential: bjx')
  # Buckingham + Ewald ionic potential
  elif(potname=='buck'):
    try: 
      from . import buck
      if(not(p==0)):
        buck.init(p)
      else:
        print("Must initialize buck with atom type defined")
        print("  use the call: pot.set('buck',p)")
        print("  where p.AtomType is set to the list of atom types")
      globals()["callBack"] = buck.force
      return force
    except:
      print('Failed to load potential: buck')
  # Art's EAM potential
  elif(potname=='eamvoter'):
#    print "in eamvoter"
    try:
      from . import eamvoter
#      print "imported"
      if(not(p==0)):
#        print "initting"
        eamvoter.init(p)
      else: 
        print("Must initialize eamvoter with atom type defined")
        print("  use the call: pot.set('eamvoter',p)")
        print("  where p.AtomType is set to the list of atom types")
      globals()["callBack"] = eamvoter.force
      return force
    except:
      print('Failed to load potential: eamvoter')
#      print "Unexpected error:", sys.exc_info()[0]
  # LEPS+HO potential (Fortran)
  elif(potname=='lepspho'):
    try:
      from . import lepspho
      globals()["callBack"] = lepspho.force
      return force
    except:
      print('Failed to load potential: lepspho')
  # NFK-PES potential (Fortran)
  elif(potname=='nfk'):
    try:
      from . import nfk
      nfk.init()
      globals()["callBack"] = nfk.force
      return force
    except:
      print('Failed to load potential: nfk')
  # Wolfe-Quapp-PES potential (Fortran)
  elif(potname=='wolfe_quapp'):
    try:
      from . import wolfe_quapp
      wolfe_quapp.init()
      globals()["callBack"] = wolfe_quapp.force
      return force
    except:
      print('Failed to load potential: wolfe_quapp')
  # dano-graeme potential (Fortran)
  elif(potname=='dano_graeme'):
    try:
      from . import dano_graeme
      dano_graeme.init()
      globals()["callBack"] = dano_graeme.force
      return force
    except:
      print('Failed to load potential: dano_graeme')
  # Lennard Jones potential (Fortran)
  elif(potname=='lj'):
    try:
      from . import lj
      lj.init()
      globals()["callBack"] = lj.force
      return force
    except:
      print('Failed to load potential: lj')
  # Lennard Jones potential two atoms (Fortran)
  elif(potname=='lj_two'):
    try:
      from . import lj_two
      lj_two.init()
      globals()["callBack"] = lj_two.force
      return force
    except:
      print('Failed to load potential: lj_two')
  # LJ potential (Python)
  elif(potname=='lj_py'):
    try: 
      from . import lj_py
      lj_py.init()
      globals()["callBack"] = lj_py.force
      return force
    except:
      print('Failed to load potential: lj_py')
  # Morse potential (Fortran)
  elif(potname=='morse'):
    try: 
      from . import morse
      morse.init()
      globals()["callBack"] = morse.force
      return force
    except:
      print('Failed to load potential: morse')
  # Morse potential (Python)
  elif(potname=='morse_py'):
    try: 
      from . import morse_py
      morse_py.init()
      globals()["callBack"] = morse_py.force
      return force
    except:
      print('Failed to load potential: morse_py')
  # Mueller-Brown potential (Fortran)
  elif(potname=='mueller'):
    try:
      from . import mueller
      globals()["callBack"] = mueller.force
      return force
    except:
      print('Failed to load potential: mueller')
  # TIP4P water potential
  elif(potname=='tip4p'):
    try: 
      from . import tip4p
      tip4p.init()
      globals()["callBack"] = tip4p.force
      return force
    except:
      print('Failed to load potential: tip4p')
  # Simple potential
  elif(potname=='simple'):
    try: 
      from . import simple
      globals()["callBack"] = simple.force
      return force
    except:
      print('Failed to load potential: simple')
  # polynomial potential
  elif(potname=='polynomial'):
    try: 
      import polynomial
      globals()["callBack"] = polynomial.force
      return force
    except:
      print('Failed to load potential: polynomial')
  # zero potential (Fortran)
  elif(potname=='zeropot'):
    try:
      from . import zeropot
      globals()["callBack"] = zeropot.force
      return force
    except:
      print('Failed to load potential: zeropot')
  # car potential (Python)
  elif(potname=='car'):
    try:
      from . import car
      globals()["callBack"] = car.force
      return force
    except:
      print('Failed to load potential: car') 
  else:
    print('No potential: ',potname) 
