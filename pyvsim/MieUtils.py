"""
.. module :: MieUtils
    :platform: Unix, Windows
    :synopsis: Mie scattering calculation
    
This module contains the Mie scattering mathematics. This was separated because
many different functions were needed to model that.

The functions were adapter from `Maetzler (Maetzler, C. Matlab functions for Mie 
scattering and absorption 
Institute of Applied Physics, University of Bern, 2002) 
<http://arrc.ou.edu/~rockee/NRA_2007_website/Mie-scattering-Matlab.pdf>`_
    
.. moduleauthor :: Ricardo Entz <maiko at thebigheads.net>

.. license::
    PyVSim v.1
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
"""
from __future__ import division
import numpy as np
import scipy.special

def Tau(pi, costheta):
    """
    Returns the value of the function :math:`\\tau`, as defined by 
    `Maetzler (Maetzler, C. Matlab functions for Mie scattering and absorption 
    Institute of Applied Physics, University of Bern, 2002) 
    <http://arrc.ou.edu/~rockee/NRA_2007_website/Mie-scattering-Matlab.pdf>`_, 
    as a function of the cosine of the
    scattering angle (:math:`\\theta`) and the :math:`\\Pi`  function:
    
    :math:`{\\tau}_n, cos(\\theta)) = n cos(\\theta) \\Pi(n) - (n+1) \\Pi(n-1)`
    
    Parameters
    ----------
    pi : numpy.array (n)
        An array containing the values of the :math:`\\Pi` function
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
    
    .. math::
    
        header(n) = {{2n+1} \\over  {n(n+1)}}
    
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
    
    .. math::
    
        \\Pi_n(cos \\theta) = {P_n^{(1)} (cos \\theta) \\over sin(\\theta)}
    
    Where:
    
    .. math::
    
        P_n^{(1)} 
        
    is a Legendre polynomial
    
    Or, in the recursive form, as shown in `Maetzler (Maetzler, C. Matlab 
    functions for Mie scattering and absorption 
    Institute of Applied Physics, University of Bern, 2002) 
    <http://arrc.ou.edu/~rockee/NRA_2007_website/Mie-scattering-Matlab.pdf>`_:
    
    .. math::
        \\Pi_n = {{2n-1} \\over {n-1}} cos(\\theta) P_{n-1} - {n \\over {n-1}} \\Pi_{n-2}
    
    With:
    
    .. math::
        \\Pi_{0} = 0
        
        \\Pi_{1} = 1  
    
    Parameters
    ----------
    nmax : int
        The number of coefficients to be calculated
        
    costheta: np.array (N)
        The angles (in radians) to be used as input to the \Pi function
        
    Returns
    -------
    Pi : generator
        A generator capable of creating an :math:`(N+1)`-sized array with the 
        values
        of the :math:`\\Pi` function

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
    Computes a matrix of Mie coefficients:
    
    .. math::
        a_n, b_n, c_n, d_n
        
    of orders :math:`n=1` to :math:`n_{max}`, complex refractive index 
    :math:`m=m^\\prime + im^{\\prime\\prime}`, 
    and size parameter 
    :math:`x=k_0\\cdot r`, 
    where 
    :math:`k_0` is the wave number in the ambient medium, 
    :math:`r` sphere radius; 
    
    Reference: `p. 100, 477 in Bohren and Huffman (1983) BEWI:TDD122
    <http://books.google.de/books/about/Absorption_and_scattering_of_light_by_sm.html?id=S1RCZ8BjgN0C&redir_esc=y>`_
    
    
    There is a limitation for the maximum allowable :math:`x` for this
    function. I have not yet verified from where it comes from
    (limit: :math:`x = 180`, which yields a maximum particle size of 
    about :math:`30\\mu m` in air)
    
    Adapted from Matlab code (Dec 2012)
    Vectorized (Apr 2013)
    
    Parameters
    ----------
    m : real or complex
        The particle refractive index.
    x : numpy.array (M)
        The Mie parameter, defined as 
        :math:`x = k_0 r = {{2  \\pi r} \\over \\lambda}`
        (where :math:`r` is the particle radius and :math:`\\lambda` is the 
        light wavelength)
        
    Returns
    -------
    (an, bn) : numpy.array (:math:`n_{max}`, M)
        The Mie :math:`a` and :math:`b` parameters calculated for :math:`n_{max}`
        factors, used to 
        calculate an approximation of the scattered light far field.
        
    Raises
    ------
    ValueError
        If the particle diameter which is being calculated forces the creation
        of too big matrices (happens when :math:`x` is higher than approximately 180)
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
    bx = scipy.special.jv(nu,x_block    ).T * np.tile(sqx,(nmax,1))
    bz = scipy.special.jv(nu,m * x_block).T * np.tile(sqz,(nmax,1))
    yx = scipy.special.yv(nu,x_block    ).T * np.tile(sqx,(nmax,1))

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
    This function calculates the Mie scattering cross section of a spherical
    particle. Its optimal use is by minimizing function calls and giving
    a particle size range instead.
    
    Parameters
    ----------
    refrativeIndex : real / complex
        The refractive index of the particles. The real part of the index
        is the classical (as used in Snell's law) and the complex is related
        to light absorption
    particleDiameters : numpy.array (N)
        A list of particle diameters to be calculated
    wavelength : real, meters
        The wavelength of the light source
    theta : numpy.array (M), radians
        A list of scattering angles to be calculated
        
    Returns
    -------
    (sigma1, sigma2) : tuple of numpy.array (M,N)
        The differential scattering cross sections for the given particle 
        diameters (column number) and scattering angles (row number). Sigma1
        stands for light that is polarized perpendicular to the propagation 
        plane, and sigma2 for light polarized parallel to the propagation plane
    
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
    """
    This function calculates the mean differential scattering cross section of 
    a particle size distribution (described by a cumulative probability
    function). 
    Note that this result shows the behavior of the distribution, which is not
    the behavior of any of the particles.
    
    Parameters
    ----------
    refrativeIndex : real / complex
        The refractive index of the particles. The real part of the index
        is the classical (as used in Snell's law) and the complex is related
        to light absorption
    particleDiameters : numpy.array (N)
        A list of particle diameters to be calculated
    percentage : numpy.array (N)
        The value of the cumulative distribution function (CDF) of the particle
        diameter distribution sampled at the points given in the
        particleDiameter array
    wavelength : real, meters
        The wavelength of the light source
    theta : numpy.array (M), radians
        A list of scattering angles to be calculated. If not given, defaults
        to a 501-element array from 0 to 180deg
         
    Returns
    -------
    (sigma1, sigma2) : tuple of numpy.array (M)
        The differential scattering cross sections for the given distribution. 
        Sigma1 stands for light that is polarized perpendicular to the 
        propagation plane, and sigma2 for light polarized parallel to the 
        propagation plane
        
    Raises
    ------
    ValueError
        If the particle diameter which is being calculated forces the creation
        of too big matrices (happens when the Mie parameter 
        (:math:`2\\pi r \\over \\lambda) is higher than approximately 180)
    """
    scs = mieScatteringCrossSections(refractiveIndex   = refractiveIndex,
                                     particleDiameters = diameters,
                                     wavelength        = wavelength, 
                                     theta             = theta)
    distribution = np.tile(percentage, (len(theta),1))
    return (np.sum(scs[0] * distribution, 1), np.sum(scs[1] * distribution, 1))
  
if __name__ == "__main__":
    import Utils
    tic = Utils.Tictoc()
    import matplotlib.pyplot as plt
    
    tic.tic()
    theta = np.linspace(0*np.pi/180,100*np.pi/180,1001)
    diam = np.arange(0.0,3.1,0.0062)
    pdf  = scipy.special.gammainc(13.9043,10.9078*diam)**0.2079
    perc = np.diff(pdf)
    diam = diam[1:]*1e-6
    scs1 = distributedSCS(1.45386, 
                          diam, 
                          perc, 
                          532e-9,
                          theta)[0]
                                                       
    tic.toc()
    apertureAngle = 0
    kernel = np.ones(apertureAngle*len(theta)/180)
    kernel = kernel / len(kernel)
    print "Kernel is %d elements long" % len(kernel)
    
    #for s in scs1:
    #    print s
    #for n,s in enumerate(perc):
    #    print diam[n], s    
        
    plt.figure(facecolor = [1,1,1])
    plt.hold(True)
    plt.plot(theta*180/np.pi, scs1, label = "Whole distribution")
    
    for n in range(0,len(diam),int(len(diam)/5.)):
        print n, diam[n]
        scs2 = distributedSCS(1.45386, 
                          np.array([diam[n]]), 
                          1, #np.array([perc[n]]), 
                          532e-9,
                          theta)[0]   
        if apertureAngle > 0:
            # filtered
            plt.plot(theta*180/np.pi,np.convolve(scs2,kernel,mode="same"), 
                     label = "D=%s micron contribution" % (diam[n]*1e6))
        else:
            # unfiltered
            plt.plot(theta*180/np.pi,scs2, 
                     label = "D=%s micron contribution" % (diam[n]*1e6))
    
    plt.xlabel("Scattering angle")
    plt.ylabel("Scattering cross section (m^2)/(sr)")
    plt.legend(loc=3)
    
    plt.yscale('log')
    plt.grid(True, which="both", axis="both")
    plt.show()