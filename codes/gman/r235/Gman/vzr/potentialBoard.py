#!/usr/bin/env python

try:
    import Gman
    from numpy import *
    import Image
    import numpy
    g2g = True
except:
    g2g = False

def potentialBoard(name, xStart, yStart, xEnd, yEnd, density, contrast):
    if globals()['g2g']:
        try:
            p = Gman.data.point()
            p.R = numpy.array([[0, 0, 0]], 'd');
            p.FreeD = numpy.array([[1, 1, 1]], 'd');
            force = Gman.pot.set(name)
            img = Image.new("RGB", (density, density), (0, 0, 0))
            val = numpy.zeros(density * density)
            width = xEnd - xStart
            height = yEnd - yStart
            stepX = width / density
            stepY = height / density
            index = 0
            maximum = -9999999999
            minimum = 9999999999
            for x in range(density):
                for y in range(density):
                    p.R[0][0] = xStart + x * stepX + 0.5 * stepX
                    p.R[0][1] = yStart + y * stepY + 0.5 * stepY
                    force(p, False)
                    val[index] = p.U
                    if p.U > maximum:
                        maximum = p.U
                    if p.U < minimum:
                        minimum = p.U
                    index = index + 1
            spread = maximum - minimum
            for i in range(len(val)):
                val[i] = ((val[i] + math.fabs(minimum)) / spread) * contrast
                if val[i] > 1:
                    val[i] = 1
            i = 0
            for x in range(density):
                for y in range(density):
                    img.putpixel((x, y), (int(256 * val[i]), 0, 128 - int(128 * val[i])))
                    i = i + 1
            return img
        except e:
            print("potentialBoard fail: ", e)
            globals()['g2g'] = False
