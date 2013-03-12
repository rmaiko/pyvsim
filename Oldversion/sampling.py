import math
import vec
import numpy as np
import Utils

coord1 = {}
coord2 = {}
c1lim = [0,1]
c2lim = [0,1]


def partition(n):
    c = vec.linspace(0,1,n)
    c2 = np.tile(c,n)
    c1 = np.zeros(len(c2))
    # print ""
    # print "c  ", c
    # print "c2 ", c2
    # print "c1 ", c1
    # print len(c2)
    for k in range(n):
        # print k*n, " to ", (k+1)*n
        # print "Put that ", np.tile(c[k],n) 
        # print "here     ", c1[k*n:(k+1)*n]
        c1[k*n:(k+1)*n] = np.tile(c[k],n) 
    return np.vstack([c1,c2]).T

def func(x):
    #return np.sum(np.sin(x),1)
    return np.sum(x * np.array([.5,3.3]) + 0.001*np.random.randn(np.size(x,0),np.size(x,1)),1)

class Interpolator2D():
    def __init__(self,
                 func,
                 points      = np.array([[0,0],[1,0],[1,1],[0,1]]),
                 calculated  = None,
                 tol=1e-6,
                 depth=0,
                 maxdepth=5):
                 
        self.points      = points
        
        if calculated = None:
            self.r           = func(points)
            
        self.M           = np.linalg.lstsq(points,r)[0]
        
        p       = self.subgrid()
        r       = func(p)
        r_est   = interpolate(p)
        
        self.err = np.sum(((r-r_est)/len(r))**2)
        
        if self.err > tol:
            self.M = None
            self.interp = 
    
    def subgrid(self):
        p = []
        for n in range(4):
            p.append((self.points(n+1 % 4) + self.points(n))/2)
        p.append((p[0]+p[2])/2)
        return p
        
    def interpolate(pts):
        if M is not None:
            return np.dot(pts,M)
        else:
            i0 = self.interp[0].interpolate(pts)
            i1 = self.interp[1].interpolate(pts)
            i2 = self.interp[2].interpolate(pts)
            i3 = self.interp[3].interpolate(pts)
            
 
    
n = 1
points = partition(n)
r      = func(points)
print ""
print points
print r
tol = 1e-8
err = 9999
print ""
while err > tol:
    M           = np.linalg.lstsq(points,r)[0]
    print M
    print len(r)
    n           = n + 1
    points      = partition(n)
    r           = func(points)
    r_est       = np.dot(points,M)
    err         = np.sum(((r-r_est)/len(r))**2)