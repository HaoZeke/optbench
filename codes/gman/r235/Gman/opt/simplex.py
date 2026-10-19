def funk(p_try):
    j=len(p_try)
    value=0
    for i in range(0, j):
      value= value + (p_try[i]-i-1)*(p_try[i]-i-1)
    return value

def amotry(p,y,psum,funk,ihi,fac):    
 # Make a flip of the polyhedron across from the highest point
 # Basically, take the average (by ndim) centers of all the degrees of freedom except those attached to the highest point and
 # Extrapolate the new point from the highest point across the geometery center of the rest of the points with a fraction of
 # the displacement between the highest point and that geometry center. (See Szabo et al : Introduction to Advanced Quantum Chemistry:...appendix)
    ndim=len(y)-1
    ihi = int(ihi)
    ptry = []
    fac1 = (1.0-fac)/float(ndim)
    fac2 = fac1 - fac
    for j in range(0,ndim):
        ptry.append(psum[j]*fac1-p[ihi][j]*fac2) 
    ytry = funk(ptry)        # Evaluate the newly extrapolate or interpolate trial point
    if ytry < y[int(ihi)]:   # If that's lower that the highest point, replace the highest point. (By this means, we get rid of the old highest point)
        y[int(ihi)] = ytry
        for j in range(0,ndim):
            psum[j] = psum[j] + ptry[j] - p[int(ihi)][j]
            p[int(ihi)][j] = ptry[j]
    return ytry


def amoeba(p,y,funk,ftol,itermax):
  tiny=1.0e-12     # This number shouldn't be too small if both abs(y[ilo]) and abs(y[ihi]) are vertually zero.
  ndim=len(y)-1
  psum = []
  for j in range(0,ndim):
    sum = 0
    for i in range(0,ndim+1):
        sum = sum + p[i][j]
    psum.append(sum)
  for count in range(0,int(itermax)):
    ilo=y.index(min(y))
    ihi=y.index(max(y))
    ytemp=y[ihi]
    y[ihi]=y[ilo]
    inhi=y.index(max(y))
    y[ihi]=ytemp
    rtol = 2.0 * abs(y[int(ihi)] - y[ilo]) /(abs(y[int(ihi)]) + abs(y[ilo]) + tiny)
    if rtol < ftol:
        temp = y[0]
        y[0] = y[ilo]
        y[ilo] = temp
        for i in range(0,ndim):
            temp = p[0][i]
            p[0][i] = p[ilo][i]
            p[ilo][i] = temp
        break
    ytry = amotry(p,y,psum,funk,ihi,-1.0)
    if ytry <= y[ilo]:                             # The new trial is even better than the lowest point. Go bolder!
        ytry = amotry(p,y,psum,funk,ihi,2.0)  # Make the polyhedron further extended along that direction.
    elif ytry >= y[inhi]:            # The trial point is still worse than the next highest point.
        ysave = y[int(ihi)]
        ytry = amotry(p,y,psum,funk,ihi,0.5)  # Do a interpolation between the trial point and the center of the rest of points.
        if ytry >= ysave:            # Geeze! The highest point (also the trial point) cannot be got rid of. Contract the polyhedron
            for i in range(0,ndim+1):
                if i != ilo:
                    for j in range(0,ndim):
                        psum[j] = 0.5*(p[i][j]+p[ilo][j])
                        p[i][j] = psum[j]
                    y[i]=funk(psum)
            psum = []
            for j in range(0,ndim):
                sum = 0
                for i in range(0,ndim+1):
                    sum = sum + p[i][j]
                psum.append(sum)
    if (count == int(itermax-1)):         # This part is not necessary.
      print('whoa whoa whoa: already ',count,' iterations\n')
      print('It did not converge. Got to save the configurations.')
      temp = y[0]
      y[0] = y[ilo]
      y[ilo] = temp
      for i in range(0,ndim):
        temp = p[0][i]
        p[0][i] = p[ilo][i]
        p[ilo][i] = temp

def mini(ndim):
    p=[]
    y=[]
    ftol=1.0e-15      # the convergence fraction.
    itermax=1000000   # the maxium number of iterations allowed.
    for i in range(0, ndim+1):
        L=[]
        for j in range(0, int(ndim)):
           L.append((j+1)*(i+5))
        p.append(L)
        y.append(funk(p[i]))
    print(p)
    print(y)
    amoeba(p,y,funk,ftol,itermax)   # the simplex optimization algorithm.
    print('The minimum configuration: p[ilo][0], p[ilo][1],....')
    print(p[0])
    print('The minimum function value at that minimum configuration is ', y[0])
    print('Well done. Good night')
