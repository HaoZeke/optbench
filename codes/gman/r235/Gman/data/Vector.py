
import math

class Vector:

    def __init__(self, elements):
        self.elements = elements[:]
        self.dimension = len(elements)

    def copy(self):
        return Vector(self.elements[:])

    def magnitude(self):
        total = 0.0
        for i in range(self.dimension):
            total = total + self.elements[i] * self.elements[i]
        return math.sqrt(total)

    def unit(self):
        mag = self.magnitude()
        elements = self.elements[:]
        for i in range(self.dimension):
            elements[i] = elements[i] / mag
        return Vector(elements)
    
