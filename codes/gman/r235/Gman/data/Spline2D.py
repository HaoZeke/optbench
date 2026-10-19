

from array import *


class Spline2D:

    def __init__(self, xa, ya):
        self.x = xa[:]
        self.y = ya[:]
        n = len(xa)
        self.A = list(range(n - 1))
        self.B = list(range(n - 1))
        self.C = list(range(n - 1))
        self.D = list(range(n - 1))
        y2 = list(range(n))
        u = list(range(n))
        y2[0] = 0.0
        u[0] = 0.0
        for i in range(1, n - 1):
            wx = xa[i + 1] - xa[i - 1]
            sig = (xa[i] - xa[i - 1]) / wx
            p = sig * y2[i - 1] + 2.0
            y2[i] = (sig - 1.0) / p
            ddydx = (ya[i + 1] - ya[i]) / (xa[i + 1] - xa[i]) - (ya[i] - ya[i - 1]) / (xa[i] - xa[i - 1])
            u[i] = (6.0 * ddydx / wx - sig * u[i - 1]) / p
        y2[n - 1] = 0.0
        for i in range(n - 2, 0, -1):
            y2[i] = y2[i] * y2[i + 1] + u[i]
        self.y2 = y2[:]
        for i in range(n - 1):
            self.A[i] = (y2[i + 1] - y2[i]) / (6 * (xa[i + 1] - xa[i]))
            self.B[i] = (y2[i] - 6 * self.A[i] * xa[i]) / 2
            self.C[i] = ya[i + 1] - ya[i] - self.A[i] * (xa[i + 1] * xa[i + 1] * xa[i + 1]) + self.A[i] * (xa[i] * xa[i] * xa[i]) - self.B[i] * (xa[i + 1] * xa[i + 1]) + self.B[i] * (xa[i] * xa[i])            
            self.C[i] /= xa[i + 1] - xa[i]
            
    def interpolate(self, x):
        n = len(self.x)
        xa = self.x
        ya = self.y
        klo = 0
        khi = n - 1
        while khi - klo > 1:
            k = (khi + klo) >> 1
            if xa[k] > x:
                khi = k
            else:
                klo = k
        h = xa[khi] - xa[klo]
        a = (xa[khi] - x) / h
        b = (x - xa[klo]) / h
        y = a * ya[klo] + b * ya[khi] + ((a * a * a - a) * self.y2[klo] + (b * b * b - b) * self.y2[khi]) * (h * h) / 6.0
        return y

#    def derivative(self, x):
#        dr = 0.000001
#        p1 = self.interpolate(x - dr)
#        p2 = self.interpolate(x + dr)
#        return (p2 - p1) / (dr * 2)

    def derivative(self, x):
        n = len(self.x)
        xa = self.x
        ya = self.y
        klo = 0
        khi = n - 1
        while khi - klo > 1:
            k = (khi + klo) >> 1
            if xa[k] > x:
                khi = k
            else:
                klo = k
        s = klo
        return 3 * self.A[s] * x * x + 2 * self.B[s] * x + self.C[s]
    
        
