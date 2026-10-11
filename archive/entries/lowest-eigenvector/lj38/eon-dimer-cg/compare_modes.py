#!/usr/bin/env python
import numpy as np
from sys import argv, exit

mode1 = np.loadtxt(argv[1]).flatten()
mode2 = np.loadtxt(argv[2]).flatten()

dp = np.abs(np.dot(mode1, mode2))
if dp >= 0.99:
    exit(0)
else:
    exit(1)
