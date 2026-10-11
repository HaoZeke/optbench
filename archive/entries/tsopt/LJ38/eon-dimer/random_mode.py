#!/usr/bin/env python
import numpy as np

mode = np.random.random((38,3))
np.savetxt("direction.dat", mode)
