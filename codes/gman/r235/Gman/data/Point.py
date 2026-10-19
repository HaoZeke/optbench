
import math
from .Vector import Vector


####################################################################################
#   Point class
####################################################################################

class Point:


####################################################################################
#   __init__ method:
#       intializes a Point object with given elements
####################################################################################

    def __init__(self, elements):
        self.elements = elements[:]
        self.dimension = len(elements)
        

####################################################################################
#   vectorTo method:
#       returns a vector from self Point to Point p2
#       assumes same dimension for each point.
####################################################################################

    def vectorTo(self, p2):
        e = []
        for i in range(self.dimension):
            e.append(p2.elements[i] - self.elements[i])
        return Vector(e)
    































