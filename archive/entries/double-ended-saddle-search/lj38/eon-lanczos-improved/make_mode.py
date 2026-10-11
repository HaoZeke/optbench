#!/usr/bin/env python
import numpy as np
from tsase.io import read_con

def pbc(r, box):
    ibox = np.linalg.inv(box)
    vdir = np.dot(r, ibox)
    vdir = (vdir % 1.0 + 1.5) % 1.0 - 0.5
    return np.dot(vdir, box)

reactant = read_con("pos.con")
displacement = read_con("displacement.con")

mode = pbc(displacement.get_positions() - reactant.get_positions(), reactant.get_cell())
mode /= np.linalg.norm(mode)

np.savetxt("direction.dat", mode)
