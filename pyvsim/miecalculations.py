"""
PyVSim 1.0
Copyright 2013 Ricardo Entz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This module contains the Mie scattering calculations for a particle or a
distribution of particles. This is not yet integrated in pyvsim
"""
from __future__ import division
import numpy as np
import scipy as sp
import scipy.special

def Tau(pi, costheta):
    """
    Returns the value of the function \Tau, as defined by Maetzler (Maetzler, C. 
    Matlab functions for Mie scattering and absorption Institute of Applied 
    Physics, University of Bern, 2002), as a function of the cosine of the
    scattering angle (theta) and the \Pi  function:
    
    \Tau(n, cos(\theta)) = n cos(\theta) \Pi(n) - (n+1) \Pi(n-1)
    
    Parameters
    ----------
    pi : numpy.array (n)
        An array containing the values of the \Pi function
    costheta : double
        The co-sine of the scattering angle
        
    Returns
    -------
    tau : numpy.array (n)
        An array containing the values of the \Theta function    
    """
    tau     = np.empty_like(pi)
    n       = np.arange(len(pi))+1
    tau[0]  = costheta
    tau[1:] = n[1:]*costheta*pi[1:] - (n[1:] + 1)*pi[:-1]
    return tau

def Header(nmax):
    """
    Generates an array with the "header", i.e. a factor to be used in the Mie
    scattering calculations. Its value is:
    
    header(n) = \over {2n+1} {n(n+1)}
    
    Parameters
    ----------
    nmax : int
        The number of factors to be calculated
        
    Returns
    -------
    header : np.array (nmax)
        The header to be used in Mie scattering calculations
    """
    n = np.arange(1,nmax+1)
    return (2*n+1)/(n*(n+1))

def Pi(nmax, costheta):
    """
    Returns a generator for the recursive calculation of the Pi function
    as defined in "Hahn, D. W. Light scattering theory Department of 
    Mechanical and Aeorospace Engineering, University of Florida, 2009", which
    is:
    
    \Pi_n(cos \theta) = P_n^{(1)} (cos \theta) / sin(theta)
    
    Where:
    
    P_n^{(1)} is a Legendre polynomial
    
    Or, in the recursive form, as shown in Maetzler (Maetzler, C. Matlab functions 
    for Mie scattering and absorption Institute of Applied Physics, University 
    of Bern, 2002):
    
    \Pi_n = \over {2n-1} {n-1} cos(\theta) P_{n-1} - \over n {n-1} \Pi_{n-2}
    
    With:
    
    \Pi_{0} = 0
    
    \Pi_{1} = 1  
    
    Parameters
    ----------
    nmax : int
        The number of coefficients to be calculated
        
    costheta: np.array (N)
        The angles (in radians) to be used as input to the \Pi function
        
    Returns
    -------
    Pi : generator
        A generator capable of creating an (N+1)-sized array with the values
        of the \Pi function

    """
    n = 2
    pi_n   = 1
    pi_n_1 = 0
    yield pi_n
    while n <= nmax+1:
        newpi = (((2*n - 1) / (n - 1)) * costheta * pi_n - 
                 (n / (n - 1)) * pi_n_1)
        pi_n_1 = pi_n
        pi_n   = newpi
        n      = n + 1
        yield newpi
        
def mie_abcd(m, x):
    """
    Computes a matrix of Mie coefficients, a_n, b_n, c_n, d_n, 
    of orders n=1 to nmax, complex refractive index m=m'+im", 
    and size parameter x=k0*a, where k0= wave number 
    in the ambient medium, a=sphere radius; 
    p. 100, 477 in Bohren and Huffman (1983) BEWI:TDD122
    C. Matzler, June 2002
    
    There is a limitation for the maximum allowable x for this
    function. I have not yet verified from where it comes from
    (limit: x = 180, which yields a maximum particle size of 
    about 30 microns)
    
    Adapted from Matlab code (Dec 2012)
    Vectorized (Apr 2013)
    """
    #
    # Parameter calculation
    # 
    nmax        = np.around(2 + np.max(x) + 4*(np.max(x)**(1/3)))
    n           = np.arange(1,nmax+1)
    nu          = n+0.5
    z           = m*x
    m2          = m*m 
    sqx         = np.sqrt(0.5*np.pi/x)
    sqz         = np.sqrt(0.5*np.pi/z)

    #
    # Bessel function calculation
    # 
    x_block = np.tile(x,(nmax,1)).T
    z_block = np.tile(z,(nmax,1)).T
    n_block = np.tile(n,(len(x),1)).T
    """
    Some matrix dimensions to help understanding the calculations
                 <--nmax-->
    x_block = [[x1, x1, x1 ...],    ^
               [x2, x2, x2 ...],  len(x)
               ...            ]]    v
               
                 <--len(x)-->
    n_block = [[1, 1, 1 ...],    ^
               [2, 2, 2 ...],  nmax(x)
               ...               | 
               [n, n, n ...]]    v               
               
                 <--nmax-->
    nu      = [nu1, nu2, nu3 ...]   1
    
                         <--len(x)-->
    bx, bz, yx  =  [[nu1.x1, nu1.x2, ... nu1.xn],   ^
                    [nu2.x1, nu2.x2, ... nu2.xn],   |
                    ...                            nmax(x)
                                                    |
                    [nun.x1, nun.x2, ... nun.xn]]   v
    bx, bz, yx have the final matrix size (FMS)
    """
    bx = sp.special.jv(nu,x_block    ).T * np.tile(sqx,(nmax,1))
    bz = sp.special.jv(nu,m * x_block).T * np.tile(sqz,(nmax,1))
    yx = sp.special.yv(nu,x_block    ).T * np.tile(sqx,(nmax,1))

    # From now on, all calculations are made with the final matrix sizes
    hx = bx+1j*yx
    
    b1x     =   np.vstack([np.sin(x)/x   ,bx[:-1,:]])
    b1z     =   np.vstack([np.sin(z)/z   ,bz[:-1,:]])
    y1x     =   np.vstack([-np.cos(x)/x  ,yx[:-1,:]])

    h1x= b1x+1j*y1x;

    ax = x_block.T * b1x - n_block*bx;
    az = z_block.T * b1z - n_block*bz;
    ahx= x_block.T * h1x - n_block*hx;

    an = (m2*bz*ax-bx*az)/(m2*bz*ahx-hx*az);
    bn = (bz*ax-bx*az)/(bz*ahx-hx*az);

    return (an,bn)

def mieScatteringCrossSections(refractiveIndex,
                               particleDiameters,
                               wavelength, 
                               theta):
    """
    XXX Auxiliary function in the calculation of Mie scattering, this could not be
    XXX yet optimized, so care should be taken
    
    Polarization parameter:
    1 = accounts for light polarized perpendicular to propagation plane
    2 = accounts for light polarized parallel to propagation plane
    None = returns average
    """
    cos_theta    = np.cos(theta)
    # Calculation of Mie parameters
    (a,b) = mie_abcd(refractiveIndex, 
                     np.pi*particleDiameters/wavelength)
    nmax         = np.size(a,0)
    nangles      = len(cos_theta)
    nd           = len(particleDiameters)
    
    pi           = np.empty((nangles,nmax,nd))
    tau          = np.empty((nangles,nmax,nd))
    header       = np.tile(Header(nmax), (nd,1)).T
    for (i, ctheta) in enumerate(cos_theta):
        pigen        = Pi(nmax, ctheta)
        pitemp       = np.fromiter(pigen, np.float)
        tautemp      = Tau(pitemp, ctheta)
        pi[i,:,:]    = np.tile(pitemp[:-1],(nd,1)).T
        tau[i,:,:]   = np.tile(tautemp[:-1], (nd,1)).T
    
    i1 = header * (a * pi + b * tau)
    i2 = header * (b * pi + a * tau)

    """
    The following sum puts the coefficients in the following form:
                <   ndiameters >
    i1,i2 =  [[i1,  i1,  ...   i1],   ^
              [i1,  i1,  ...   i1],  nangles
              ...
              [i1,  i1,  ...   i1]]   v
    so, each particle diameter is represented by a column
    """
    i1 = np.sum(i1,1)
    i2 = np.sum(i2,1)

    i1 = (np.abs(i1)**2) * (1/4) * (wavelength/np.pi)**2
    i2 = (np.abs(i2)**2) * (1/4) * (wavelength/np.pi)**2
    
    return (i1, i2)

def distributedSCS(refractiveIndex, 
                   diameters, 
                   percentage, 
                   wavelength,
                   theta = np.linspace(0,np.pi,501)):
    scs = mieScatteringCrossSections(refractiveIndex   = refractiveIndex,
                                     particleDiameters = diameters,
                                     wavelength        = wavelength, 
                                     theta             = theta)
    distribution = np.tile(percentage, (len(theta),1))
    return (np.sum(scs[0] * distribution, 1), np.sum(scs[1] * distribution, 1))
  

import Utils
theta = np.linspace(0,np.pi,501)
tic = Utils.Tictoc()
#tic.reset()
#scs = mieScatteringCrossSections(refractiveIndex   = 1.4,
#                                 particleDiameters = np.array([1700, 170, 17])*1e-9,
#                                 wavelength        = 532e-9, 
#                                 theta             = theta)
#tic.toc()
#scs = scs[0]

diam = np.arange(0.1,3.1,0.1)*1e-6

perc = np.array([0.0058,
                 0.0293,
                 0.0569,
                 0.0797,
                 0.0946,
                 0.1015,
                 0.1015,
                 0.0961,
                 0.0871,
                 0.0762,
                 0.0644,
                 0.0528,
                 0.0421,
                 0.0325,
                 0.0245,
                 0.0179,
                 0.0127,
                 0.0087,
                 0.0059,
                 0.0038,
                 0.0024,
                 0.0015,
                 0.0009,
                 0.0005,
                 0.0003,
                 0.0002,
                 0,
                 0,
                 0,
                 0])
import matplotlib.pyplot as plt
tic.reset()
scs1 = distributedSCS(1.45386, 
                      diam, 
                      perc, 
                      532e-9,
                      theta)[0]
scs2 = distributedSCS(1.45386, 
                      np.array([diam[10]]), 
                      np.array([perc[10]]), 
                      532e-9,
                      theta)[0]   
scs3 = distributedSCS(1.45386, 
                      diam[:10], 
                      perc[:10], 
                      532e-9,
                      theta)[0]     
scsone = mieScatteringCrossSections(refractiveIndex   = 1.45386,
                                 particleDiameters = np.array([1])*1e-6,
                                 wavelength        = 532e-9, 
                                 theta             = theta)[0]                                                     
tic.toc()

#for s in scs1:
#    print s
plt.figure(facecolor = [1,1,1])
plt.hold(True)
plt.plot(theta*180/np.pi,scs1,      label = "Whole distribution")
for n in [7,8,9,10,11,12,13]:
    print n, diam[n]
    scs2 = distributedSCS(1.45386, 
                      np.array([diam[n]]), 
                      np.array([perc[n]]), 
                      532e-9,
                      theta)[0]   
    plt.plot(theta*180/np.pi,scs2, 
             label = "D=%s micron contribution" % (diam[n]*1e6))

plt.xlabel("Scattering angle")
plt.ylabel("Scattering cross section (m^2)/(sr)")
plt.legend()

#l1 = plt.plot(theta*180/np.pi, scs[:,0]*1e-3*1e4, 'k', label="d = 1.7e-6m * 1e-3")
#l2 = plt.plot(theta*180/np.pi, scs[:,1]*1e4,      'r', label="d = 170e-9m")
#l3 = plt.plot(theta*180/np.pi, scs[:,2]*1e6*1e4,  'b', label="d = 17e-9m * 1e6")
#plt.legend()
#plt.xlabel("Scattering angle (deg)")
#plt.ylabel("Diff. Scattering Cross Sec (cm^2sr^{-1})")
#plt.title("compare with: 'Light Scattering Theory' by David W Hahn, \n m = 1.4, lambda = 532nm")    
plt.yscale('log')
#plt.ylim(1e-14,1e-10)
plt.grid(True, which="both", axis="both")
plt.show()