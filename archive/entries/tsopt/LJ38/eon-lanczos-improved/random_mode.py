#!/usr/bin/env python
import numpy as np
from sys import argv

np.random.seed(int(argv[1]))
mode = np.random.random((38,3))
np.savetxt("direction.dat", mode)
