try:
    from OpenGL.GL import *
    from OpenGL.GLUT import *
    from OpenGL.GLU import *
    import Image
    import threading
    import profile
    import array
    import time
    g2g = True
except:
    g2g = False
    print("vzr fail")

class Vzr:

    idSphere = 1


######################################################################################################
######################################################################################################


    def __init__(self,width=512, height=512):
        if globals()["g2g"]:
            try:
                self.width= width
                self.height = height
                self.updated = True
                self.clearBit = True
                self.clearRed = 0.0
                self.clearGreen = 0.0
                self.clearBlue = 0.0
                self.camX = 0
                self.camY = 0
                self.camZ = 5
                self.lookX = 0
                self.lookY = 0
                self.lookZ = 0
                self.upX = 0
                self.upY = 1
                self.upZ = 0
                self.sphereList = []
                self.lineList = []
                self.boardList = []
                self.killed = False
                threading.Thread(target=self.run).start()
            except:
                globals()["g2g"] = False
        

######################################################################################################
######################################################################################################


    def kill(self):
        if globals()["g2g"]:
            try:
                self.kill = True
            except:
                globals()["g2g"] = False
        

######################################################################################################
######################################################################################################


    def run(self):
        if globals()["g2g"]:
            try:
                glutInit(())
                glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
                glutInitWindowSize(self.width, self.height)
                self.window = glutCreateWindow("Visualizer")
                glutDisplayFunc(self.draw)
                glutIdleFunc(self.draw)
                glutReshapeFunc(self.resize)
                glClearDepth(1.0)
                glShadeModel(GL_SMOOTH)
                glDepthFunc(GL_LEQUAL)
                glEnable(GL_DEPTH_TEST)
                glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
                glLightfv(GL_LIGHT1, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
                glLightfv(GL_LIGHT1, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
                glLightfv(GL_LIGHT1, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
                glEnable(GL_LIGHT1)
                glEnable(GL_LIGHTING)
                glEnable(GL_NORMALIZE)
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(45.0, float(512)/float(512), 0.1, 100000.0)
                glMatrixMode(GL_MODELVIEW)
                glNewList(Vzr.idSphere, GL_COMPILE)
                glutSolidSphere(1, 20, 20)
                glEndList()
                glutMainLoop()
            except:
                globals()["g2g"] = False

######################################################################################################
######################################################################################################


    def resize(self, width, height):
        if globals()["g2g"]:
            try:
                if height == 0:
                    height = 1
                glViewport(0, 0, width, height)
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(45.0, float(width)/float(height), 0.1, 100000.0)
                glMatrixMode(GL_MODELVIEW)
            except:
                globals()["g2g"] = False


######################################################################################################
######################################################################################################


    def update(self):
        if globals()["g2g"]:
            try:
                self.updated = False
                while not self.updated:
                    time.sleep(0.001)
            except:
                globals()["g2g"] = False
        
        
######################################################################################################
######################################################################################################

    def draw(self):
        if globals()["g2g"]:
            try:
                if self.updated:
                    time.sleep(0.001)
                    return
                if self.clearBit:
                    glClearColor(self.clearRed, self.clearGreen, self.clearBlue, 0.0)
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    self.clearBit = False
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                gluLookAt(self.camX, self.camY, self.camZ, self.lookX, self.lookY, self.lookZ, self.upX, self.upY, self.upZ)
                glLightfv(GL_LIGHT1, GL_POSITION, [500.0, 500.0, 1000.0, 1.0])

                self.drawBoards()
                self.drawSpheres()
                self.drawLines()
                glutSwapBuffers()
                self.updated = True
            except:
                globals()["g2g"] = False


######################################################################################################
######################################################################################################


    def clear(self, red, green, blue):
        if globals()["g2g"]:
            try:
                self.clearBit = True
                self.clearRed = red
                self.clearGreen = green
                self.clearBlue = blue
            except:
                globals()["g2g"] = False


######################################################################################################
######################################################################################################


    def setCamera(self, x, y, z, lx, ly, lz, ux, uy, uz):
        if globals()["g2g"]:
            try:
                self.camX = x
                self.camY = y
                self.camZ = z
                self.lookX = lx
                self.lookY = ly
                self.lookZ = lz
                self.upX = ux
                self.upY = uy
                self.upZ = uz
            except:
                globals()["g2g"] = False

######################################################################################################
######################################################################################################


    def drawSphere(self, x, y, z, radius, r, g, b):
        if globals()["g2g"]:
            try:
                self.sphereList.append([x, y, z, radius, r, g, b])
            except:
                globals()["g2g"] = False

    def drawSpheres(self):
        if globals()["g2g"]:
            try:
                glMaterialf(GL_FRONT, GL_SHININESS, 128.0)
                glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0])
                glMaterialfv(GL_FRONT, GL_AMBIENT, [0.0, 0.0, 0.0])
                while len(self.sphereList) > 0:
                    s = self.sphereList.pop()
                    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [s[4], s[5], s[6]])
                    glLoadIdentity()
                    gluLookAt(self.camX, self.camY, self.camZ, self.lookX, self.lookY, self.lookZ, self.upX, self.upY, self.upZ)
                    glTranslated(s[0], s[1], s[2])
                    glScaled(s[3], s[3], s[3])
                    glCallList(Vzr.idSphere)
            except:
                globals()["g2g"] = False


######################################################################################################
######################################################################################################

    def drawLine(self, x1, y1, z1, x2, y2, z2, r, g, b):
        if globals()["g2g"]:
            try:
                self.lineList.append([x1, y1, z1, x2, y2, z2, r, g, b])
            except:
                globals()["g2g"] = False

    def drawLines(self):
        if globals()["g2g"]:
            try:
                glDisable(GL_LIGHTING)
                glLoadIdentity()
                gluLookAt(self.camX, self.camY, self.camZ, self.lookX, self.lookY, self.lookZ, self.upX, self.upY, self.upZ)
                glBegin(GL_LINES)
                while len(self.lineList) > 0:
                    s = self.lineList.pop()
                    glColor3f(s[6], s[7], s[8])
                    glVertex3f(s[0], s[1], s[2])
                    glVertex3f(s[3], s[4], s[5])
                glEnd()
                glEnable(GL_LIGHTING)
            except:
                globals()["g2g"] = False

######################################################################################################
######################################################################################################

    def drawBoard(self, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, img):
        if globals()["g2g"]:
            try:
                self.boardList.append([x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, img])
            except:
                globals()["g2g"] = False

    def drawBoards(self):
        if globals()["g2g"]:
            try:
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
                glEnable(GL_TEXTURE_2D)
                glDisable(GL_LIGHTING)
                while len(self.boardList) > 0:
                    b = self.boardList.pop()
                    im = b[12].tostring("raw", "RGBX", 0, -1)
                    glPixelStorei(GL_UNPACK_ALIGNMENT,1)
                    glTexImage2D(GL_TEXTURE_2D, 0, 4, b[12].size[0], b[12].size[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, im)
                    glBegin(GL_QUADS)
                    glTexCoord2f(0, 0)
                    glVertex3f(b[0], b[1], b[2])
                    glTexCoord2f(1, 0)
                    glVertex3f(b[3], b[4], b[5])
                    glTexCoord2f(1, 1)
                    glVertex3f(b[6], b[7], b[8])
                    glTexCoord2f(0, 1)
                    glVertex3f(b[9], b[10], b[11])
                    glEnd()
                glEnable(GL_LIGHTING)
                glDisable(GL_TEXTURE_2D)
            except:
                globals()["g2g"] = False
        

######################################################################################################
######################################################################################################


######################################################################################################
######################################################################################################




