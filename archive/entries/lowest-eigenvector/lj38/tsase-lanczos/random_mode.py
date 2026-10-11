#!/usr/bin/env python
import numpy as np

np.random.seed(314159)
mode = np.random.random((38,3))
mode /= np.linalg.norm(mode)
np.savetxt("direction.dat", mode)
