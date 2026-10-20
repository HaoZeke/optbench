#!/usr/bin/env python

try:
    from Gman import *
    import Gman.opt as Gopt
    import Gman.min as Gmin
    from numpy import *
    from .vzr import Vzr
    import numpy
    import Image
    g2g = True
except:
    g2g = False

def lepsBoard(xStart, yStart, xEnd, yEnd, density, contrast):
    if globals()['g2g']:
        try:
            p = data.point()
            p.R = numpy.array([[0, 0]], 'd')
            p.FreeD = numpy.array([[1, 1]], 'd')
            force = pot.set('lepspho')
            img = Image.new("RGB", (density, density), (0, 0, 0))
            val = zeros(density * density)
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
            print(e)
            globals()['g2g'] = False

# Testing, testing, 1, 2, 3...
'''v = Vzr()
v.setCamera(1.9, -0.3, 5, 1.9, -0.3, 0, 0, 1, 0)

img = lepsBoard(0.2, -2.0, 3.6, 1.4, 128, 20)

while True:
    v.clear(0, 0, 0)
    v.drawBoard(0.2, 1.4, 0, 3.6, 1.4, 0, 3.6, -2.0, 0, 0.2, -2.0, 0, img)
    v.update()'''

