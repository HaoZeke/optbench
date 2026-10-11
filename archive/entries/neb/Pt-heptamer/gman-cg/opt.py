#!/usr/bin/env python

import time

import Gman.neb as      Gneb
import Gman.opt as      Gopt
from   Gman     import  *
from   Gman.pot import  *

p1=data.point()
p2=data.point()

mina = ['min01/a.con', 'min02/a.con', 'min03/a.con', 'min04/a.con', 'min05/a.con', 'min06/a.con', 'min07/a.con', 'min08/a.con', 'min09/a.con', 'min10/a.con', 'min11/a.con', 'min12/a.con', 'min13/a.con'] 
minb = ['min01/b.con', 'min02/b.con', 'min03/b.con', 'min04/b.con', 'min05/b.con', 'min06/b.con', 'min07/b.con', 'min08/b.con', 'min09/b.con', 'min10/b.con', 'min11/b.con', 'min12/b.con', 'min13/b.con'] 

potential=pot.set('morse')

forceCalls = []

for i in range(13):

    #opt=Gopt.lbfgs()
    #opt.init(maxmove=0.2, alpha=0.05, memory=50, min='line')

    #opt = Gopt.lbfgs()
    #opt.init(maxmove = 0.2, alpha = 0.05, memory = 25, min = 'line')

    #opt=Gopt.glbfgs()
    #opt.init(maxmove=0.2, min='line', alpha=0.05, memory=50)

    #opt=Gopt.glbfgs()
    #opt.init(maxmove=0.2, min='hess', alpha=0.05, memory=25)

    #opt=Gopt.qm()
    #opt.init(dt=0.01, maxmove = 0.2)

    #opt=Gopt.fire()
    #opt.init(maxmove = 0.2, dt = 0.15)

    #opt=Gopt.sd()
    #opt.init(stepR=0.02, maxmove=0.2)

    opt=Gopt.cg()
    opt.init(maxmove=0.2, dR=0.0001)

    #opt=Gopt.rk4()
    #opt.init()

    io.loadcon(p1, "minima/"+mina[i])
    io.loadcon(p2, "minima/"+minb[i])
    neb = Gneb.neb()
    neb.new(p1, p2, images = 8)
    neb.relax(potential, opt.step, k=5.0, tangent="new", itrmax = 1000, ppd = 1, fmax = 0.01, fsaddle = 'off', method = "ci", converge="Fmag")
    
    forceCalls.append(pot.count()/8.0)
    pot.countReset()


f = open("benchmark.dat", 'w')
f.write("force_calls %d\n" % (sum(forceCalls)/13.0))
f.write("force_calls_max %d\n" % max(forceCalls))
f.write("force_calls_min %d\n" % __builtins__.min(forceCalls))
f.write("algorithm lbfgs\n")
f.write("code Gman\n")
f.write("code_version r235\n")
f.write("date %s\n" % time.ctime())
f.write("contributor Rye Terrell\n")
f.close()
