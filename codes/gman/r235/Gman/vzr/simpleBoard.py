#!/usr/bin/env python

from Gman import *
import Gman.opt as Gopt
import Gman.min as Gmin
from numpy import *
from .vzr import Vzr
import Image


def simpleBoard(xStart, yStart, xEnd, yEnd, density, contrast):
    p=data.point()
    io.loadcon(p,'simplemina.con')
    force=pot.set('simple')
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
            force(p)
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
    print(val)
    i = 0
    for x in range(density):
        for y in range(density):
            img.putpixel((x, y), (int(256 * val[i]), int(256 * val[i]), int(256 * val[i])))
            i = i + 1
    return img

# Testing, testing, 1, 2, 3...
'''v = Vzr()
v.setCamera(0, 0.5, 5, 0, 0.5, 0, 0, 1, 0)

img = muellerBoard(-1.5, -0.5, 1.25, 1.75, 64, 1)
print "done board"

#while True:
v.clear(0, 0, 0)
    #v.drawBoard(-1.5, 1.75, 0, 1.25, 1.75, 0, 1.25, -0.5, 0, -1.5, -0.5, 0, img)
#v.update()'''

