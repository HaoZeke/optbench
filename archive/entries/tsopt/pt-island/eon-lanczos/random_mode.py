#!/usr/bin/env python
import numpy as np
from tsase.io import read_con

reactant = read_con("pos.con")
displacement = read_con("displacement.con")

mode = displacement.get_positions() - reactant.get_positions()
mode /= np.linalg.norm(mode)

np.savetxt("direction.dat", mode)
