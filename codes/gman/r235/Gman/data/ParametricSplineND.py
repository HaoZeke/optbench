
from .Spline2D import *


class ParametricSplineND:


    def __init__(self):
        self.splines = []
        self.count = 0

    def addDimension(self, parameter, value):
        self.splines.append(Spline2D(parameter, value))
        self.count += 1

    def pointAtLength(self, t):
        p = list(range(self.count))
        for i in range(self.count):
            p[i] = self.splines[i].interpolate(t)
        return p
        
    def derivativeAtLength(self, t):
        p = list(range(self.count))
        for i in range(self.count):
            p[i] = self.splines[i].derivative(t)
        return p
        
        

    
