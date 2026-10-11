import numpy as np
import os

from pele.transition_states import findTransitionState

def findTS(coords, pot):
    print "using this findTS", os.path.abspath(__file__)
    np.random.seed(0)
    ''' routine to execute a single transition state refinement for the benchmark ''' 
    lowestEigenvectorQuenchParams = dict(nsteps=46, 
                                         tol=0.68,
                                         maxstep=.2,
                                         first_order=False,
                                         iprint=-10,
                                         H0=0.1,
                                         )
    return findTransitionState(coords, pot, tol=1e-3/np.sqrt(3.*38.), 
                               lowestEigenvectorQuenchParams=lowestEigenvectorQuenchParams,
                               tangentSpaceQuenchParams=dict(
                                                             maxstep=.2,
                                                             M=4,
                                                             H0=.1,
                                                             ),
                               demand_initial_negative_vec=False,
                               nsteps_tangent1=14, 
                               nsteps_tangent2=16, 
                               nfail_max=200,
                               nsteps=1001,
                               max_uphill_step=0.5,
                               max_uphill_step_initial=0.2,
                               verbosity=10,
                               iprint=1,
                               check_negative=False,
                               )
