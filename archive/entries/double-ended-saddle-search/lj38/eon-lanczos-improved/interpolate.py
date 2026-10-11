#!/usr/bin/env python
from aselite import read_con, write_con
from sys import argv
import numpy as np

atoms1 = read_con(argv[1])
atoms2 = read_con(argv[2])

def pbc(r, box):
    ibox = np.linalg.inv(box)
    vdir = np.dot(r, ibox)
    vdir = (vdir % 1.0 + 1.5) % 1.0 - 0.5
    return np.dot(vdir, box)

v = pbc(atoms2.positions-atoms1.positions, atoms1.cell)
atoms1.positions += v/2.0

write_con(argv[3], atoms1)
