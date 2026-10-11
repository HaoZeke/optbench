import numpy as np

from pele.systems import LJCluster
from pele.potentials import LJ
from pele.utils.xyz import read_xyz
from pele.utils.hessian import sort_eigs
from pele.transition_states import findLowestEigenVector

class PotWrapper():
    """a LJ potential wrapper to count the number of function calls"""
    def __init__(self, pot):
        self.pot = pot
        self.ncalls = 0
    def getEnergy(self, coords):
        self.ncalls += 1
        return self.pot.getEnergy(coords)
    def getEnergyGradient(self, coords):
        self.ncalls += 1
        return self.pot.getEnergyGradient(coords)

class OverlapStopCrit(object):
    def __init__(self, vec):
        self.vec = vec.copy()
        norm = np.linalg.norm(self.vec)
#        print "norm vec", norm
#        print self.vec
        self.vec /= np.linalg.norm(self.vec)
    
    def __call__(self, coords=None, tol=None, **kwargs):
        norm = np.linalg.norm(coords)
        overlap = self.vec.dot(coords) / norm
        overlap = np.abs(overlap)
#        print overlap, norm
        if np.isnan(overlap):
            print "overlap is nan"
            print coords
            print self.vec
            raise Exception()
        return overlap > tol

def main(outfile="results.dat"):
    overlap_tol = 0.99
    with open(outfile, "w") as fout:
        fout.write("#nfev success eval eval_exact overlap rms\n")
        for i in range(200):
            ret, eval, evec, nfev = do_one_run(i)
            
            overlap = np.abs(evec.dot(ret.eigenvec) / np.linalg.norm(ret.eigenvec))
            success = overlap > overlap_tol
            fout.write("%d %d %f %f %f %f\n" % (nfev, int(success), ret.eigenval, eval, overlap, ret.rms))
        

def do_one_run(i):
    np.random.seed(0)
    natoms = 38
    system = LJCluster(natoms)
    pot = PotWrapper(LJ())
    kwargs = system.params.double_ended_connect.local_connect_params.tsSearchParams.lowestEigenvectorQuenchParams
    orthogZeroEigs = system.get_orthogonalize_to_zero_eigenvectors()
    
    overlap_tol = 0.99
    
    datadir = "./coords"
    
    # get coordinates
    fname = datadir + "/%04d.xyz" % i
    xyz = read_xyz(open(fname, "r"))
    coords = xyz.coords.flatten()
    
    # get correct answer
    mode_fname = datadir + "/%04d_mode" % i
    mode = np.genfromtxt(mode_fname)
    with open(mode_fname, "r") as fin:
        line = fin.readline()
        eval = float(line.split()[1])
    
    stop_crit = OverlapStopCrit(mode)
    kwargs["alternate_stop_criterion"] = stop_crit
    
    kwargs["tol"] = overlap_tol
    kwargs["M"] = 100
    kwargs["maxstep"] = 2.
    kwargs["first_order"] = True

    eigenvec0 = np.genfromtxt(datadir + "/initial_mode")

    ret = findLowestEigenVector(coords, pot, orthogZeroEigs=orthogZeroEigs, iprint=100, eigenvec0=eigenvec0, **kwargs)
    assert stop_crit(coords=ret.eigenvec, tol=overlap_tol)
    return ret, eval, mode, pot.ncalls
        

if __name__ == "__main__":
    main()


