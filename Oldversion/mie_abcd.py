from __future__ import division
import vtk
import numpy as np
import math
import copy
import Utils
import vec
from pprint import pprint


def besselj(nu,z):
    err = 1
    oldsum = 0
    sum = 0
    k = 0
    
    while err > 1e-8:
        sum = sum + ((-1)**k)*((0.5*z)**(2*k+nu))/(math.factorial(k)*math.gamma(nu+k+1))
        err     = np.abs(oldsum - sum)
        oldsum  = sum
        k       = k + 1
    return sum
    
def bessely(nu,z):
    return (besselj(nu,z)*math.cos(math.pi*nu)-besselj(-nu,z))/math.sin(nu*math.pi)


x = 1e-6 * 2 * np.pi / (532e-9)
m = 1.333 + 0.00001*1j
def mie_abcd(m, x)
    #
    # Parameter calculation
    # 
    nmax        = int(np.round(2+x+4*x**(1/3)))
    n           = np.array(range(1,nmax+1))
    nu          = n+0.5
    z           = m*x
    m2          = m*m 
    sqx         = np.sqrt(0.5*np.pi/x)
    sqz         = np.sqrt(0.5*np.pi/z)

    #
    # Bessel function calculation
    # 
    bx = []
    yx = []
    bz = []

    for nup in nu:
        bx.append(besselj(nup, x)*sqx)
        bz.append(besselj(nup, z)*sqz)
        yx.append(bessely(nup, x)*sqx)
    bx = np.array(bx)
    yx = np.array(yx)
    bz = np.array(bz)
        
    hx = bx+1j*yx

    b1x     =   np.hstack([np.sin(x)/x   ,bx[:-1]])
    b1z     =   np.hstack([np.sin(z)/z   ,bz[:-1]])
    y1x     =   np.hstack([-np.cos(x)/x  ,yx[:-1]])

    h1x= b1x+1j*y1x;

    ax = x*b1x-n*bx;

    az = z*b1z-n*bz;

    ahx= x*h1x-n*hx;

    an = (m2*bz*ax-bx*az)/(m2*bz*ahx-hx*az);

    bn = (bz*ax-bx*az)/(bz*ahx-hx*az);

    cn = (bx*ahx-hx*ax)/(bz*ahx-hx*az);

    dn = m*(bx*ahx-hx*ax)/(m2*bz*ahx-hx*az);

    return (np.vstack([an,bn,cn,dn]).T)
