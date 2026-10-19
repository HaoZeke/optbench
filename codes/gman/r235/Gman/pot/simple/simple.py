import numpy
import math

def force(p):
    Ax = -1.0
    Ay = -1.0
    x = p.R[0][0]
    y = p.R[0][1]
    p.U = Ax * math.cos(2 * math.pi * x) + Ay * math.cos(2 * math.pi * y)
    p.F = p.R * 0.0
    p.F[0][0] = 2 * Ax * math.pi * math.sin(2 * math.pi * x)
    p.F[0][1] = 2 * Ay * math.pi * math.sin(2 * math.pi * y)
