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

def average_square_scattering_angle(
            theta : np.ndarray,
            P     : np.ndarray) -> np.ndarray:
    ''' Calculates the RMS averaged scattering angle of a phase function
    
    The calulation uses the following formula:
    
    .. math:
    \left< \theta^2\right> = \int_{-\pi}^{+\pi} \int_{0}^{2\pi}  
        P \left( \theta \right ) \theta^2 sin(\theta) d\varphi d\theta 
        
    Which is the integral over the sphere of the scattering angle. 
    Unpolarized light is assumed, so that the phase function depends only
    on $\theta$.
    
    :param theta: numpy.array (M)
        List of angles
    :param P: numpy.array (N, M)
        Value of phase function at theta. Each row is a different phase
        function
        
    :return numpy.array (N)
        RMS average scattering angle for each phase function
    '''
    return integral_over_sphere(theta, P*theta**2)

def asymmetry_parameter(
            theta : np.ndarray,
            P     : np.ndarray) -> np.ndarray:
    ''' Calculate the anisotropy of a list of 2D polar phase function 
    
    The following formula is used:
    
    .. math:
    g = \int_{-\pi}^{+\pi} \int_{0}^{2\pi}  
        P \left( \theta \right ) cos(\theta) sin(\theta) d\varphi d\theta 
         
    Unpolarized light is assumed, so that the phase function depends only
    on $\theta$.    
    
    :param theta: numpy.array (M) 
        Deflection angles
    :param P: numpy.array (N, M) 
        Value of phase function at theta. Each row of N is a different phase
        function
    
    :return numpy.array (N)
        The anisotropy parameter, g for each phase function
    '''
    return integral_over_sphere(theta, P*np.cos(theta))

def integral_over_sphere(theta  : np.ndarray, 
                         f      : np.ndarray) -> np.ndarray:
    ''' Calculates the integral of a polar function over a 3D sphere
    
    This function does NOT execute resampling, so accuracy depends on input
    granularity
    
    The following formulation is used:
    
    .. math:
    \int_{-\pi}^{+\pi} \int_{0}^{2\pi}  
        F(\theta) sin(\theta) d\varphi d\theta  
    
    :param theta: numpy.array (M) 
        Deflection angles, must span [-pi,pi] range for integral over 
        whole sphere
    :param f: numpy.array (N, M)    
        Values of the functions at the given angles. Each row is a 
        different function
    
    :return numpy.array (N)
        The value of the integral
    '''
    if len(f.shape) < 2:
        f = np.reshape(f,(1,-1))
        
    return (np.trapz(np.einsum("ij,j->ij", f, np.sin(theta)), 
                     theta,
                     axis = 1) * 
            2*np.pi)

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
        
def j_n(n, z):
    """ Spherical Bessel function of first kind of order n
    
    Parameters
    ----------
    n    :    numpy.array(M,N)
        Order of the function
    z    :    numpy.array(M,N)
        Function argument
        
    Returns
    -------
    jn(z)    : numpy.array(M,N)
        Function value
    """    
    return np.sqrt(np.pi / (2*z)) * scipy.special.jv(n + 0.5, z)

def y_n(n, z):
    """ Spherical Bessel function of second kind of order n
    
    Parameters
    ----------
    n    :    numpy.array(M,N)
        Order of the function
    z    :    numpy.array(M,N)
        Function argument
        
    Returns
    -------
    yn(z)    : numpy.array(M,N)
        Function value
    """
    return np.sqrt(np.pi / (2*z)) * scipy.special.yv(n + 0.5, z)

def h_n(n, z):
    """ Spherical Hankel function of order n
    
    Parameters
    ----------
    n    :    numpy.array(M,N)
        Order of the function
    z    :    numpy.array(M,N)
        Function argument
        
    Returns
    -------
    yn(z)    : numpy.array(M,N)
        Function value
    """
    return j_n(n, z) + 1j*y_n(n, z)

def phase_function_normalize(
            theta : np.ndarray,
            P     : np.ndarray) -> np.ndarray:
    ''' Normalizes a phase function to a probability density function
    
    This function normalizes the phase function by the value of 
    its integral over a sphere:
    
    .. math:
        f = \int_{-\pi}^{+\pi} \int_{0}^{2\pi}  
                F(\theta) sin(\theta) d\varphi d\theta
                
    .. math:            
        \overline{F} = F / f
    
    :param theta: numpy.array (M)
        List of angles
    :param P: numpy.array (N, M)
        Value of phase function at theta. Each row is a different phase
        function
        
    :return numpy.array (N, M)
        Probability density function for each of the N phase 
        functions
    '''
    return np.einsum("ij,i->ij", P, 1 / integral_over_sphere(theta, P))

def phase_function_to_pdf_cdf(theta, P):
    ''' Calculates the probability density function (PDF) and cumulative
    density function (CDF) for scattering based on a phase function
    
    :param theta: numpy.array (M)
        List of angles
    :param P: numpy.array (N, M)
        Value of phase function at theta. Each row is a different 
        phase function
        
    :return PDF, CDF : numpy.array (N, M)
        Distribution functions for each of the N phase 
        functions
    '''
    pdf  = phase_function_normalize(theta, P)
    
    delta_theta     =   (theta[1:] - theta[:-1])
    sin_theta       =   np.sin(theta)
    CDF             =   np.zeros_like(pdf)
    
    for i in range(pdf.shape[0]): # For each row
        p_plus          = (pdf[i][1: ]*sin_theta[1: ] + 
                           pdf[i][:-1]*sin_theta[:-1])/2
        trapz           = p_plus*delta_theta
        CDF[i][:-1]     = np.cumsum(trapz)*2*np.pi
        CDF[i][-1]      = 1
        
    return (pdf, CDF)  
 
def mie_abcd(m, 
             x,
             mu_sphere = 1.25663753e-6,
             mu_medium = 1.25662700e-6):
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
    (an, bn) : numpy.array (M, :math:`n_{max}`)
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
    z           = m*x
    mu1         = mu_sphere / mu_medium

    X  = np.tile(np.reshape(x, (-1,1)),(1,nmax))
    Z  = np.tile(np.reshape(z, (-1,1)),(1,nmax))
    N  = np.tile(n, (len(x),1))

    """
    Some matrix dimensions to help understanding the calculations
    
    nrows : diameters
    ncols : nmax
    """
    
    #
    # Bessel function calculation
    # 
    j_N_X  = j_n(N,   X) #scipy.special.jv(nu,x_block    ).T * np.tile(sqx,(nmax,1))
    j_N_MX = j_n(N, m*X) #scipy.special.jv(nu,m * x_block).T * np.tile(sqz,(nmax,1))
    y_N_X  = y_n(N,   X) #scipy.special.yv(nu,x_block    ).T * np.tile(sqx,(nmax,1))

    # Hankel function
    h_N_X = j_N_X + 1j * y_N_X
    
    #
    # Calculate the derivatives of functions using the formula:
    #
    # (x*f(n,x))' = x*f(n-1,x) + n*f(n,x) 
    #
    
    # Calculating f(n-1,x) f_(N-1)_X
    j_N1_X  = np.empty_like(j_N_X, dtype = np.complex)
    j_N1_MX = np.empty_like(j_N_X, dtype = np.complex)
    y_N1_X  = np.empty_like(j_N_X, dtype = np.complex)
    
    j_N1_X[:,1:]  = j_N_X[:,:-1]
    j_N1_MX[:,1:] = j_N_MX[:,:-1]
    y_N1_X[:,1:]  = y_N_X[:,:-1]
    
    # Boundary conditions for (f(0,x))'
    j_N1_X[:,0]  =  np.sin(x)/x
    j_N1_MX[:,0] =  np.sin(z)/z
    y_N1_X[:,0]  = -np.cos(x)/x  
    
    # Back 1 Hankel function 
    h_N1_X = j_N1_X + 1j*y_N1_X;

    # Derivative of functions
    X_j_N_X_prime    = X * j_N1_X  - N*j_N_X;
    MX_j_N_MX_prime  = Z * j_N1_MX - N*j_N_MX;
    X_h_N_X_prime    = X * h_N1_X  - N*h_N_X;

    a_N = ((m*m * j_N_MX * X_j_N_X_prime) - (mu1 * j_N_X * MX_j_N_MX_prime)) /\
          ((m*m * j_N_MX * X_h_N_X_prime) - (mu1 * h_N_X * MX_j_N_MX_prime))
          
    b_N = ((mu1 * j_N_MX * X_j_N_X_prime) - (j_N_X * MX_j_N_MX_prime)) /\
          ((mu1 * j_N_MX * X_h_N_X_prime) - (h_N_X * MX_j_N_MX_prime))
    
    a_N[np.isnan(a_N)] = 0
    b_N[np.isnan(b_N)] = 0

    return (a_N,b_N)

def mie_characteristics(
        refractiveIndex,
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
    particleDiameters : numpy.ndarray (N)
        A list of particle diameters to be calculated
    wavelength : real, meters
        The wavelength of the light source
    theta : numpy.ndarray (M), radians
        A list of scattering angles to be calculated
        
    Returns
    -------
    (S1, S2, P11, Q_sca, Q_abs, Q_ext) : tuple of numpy.ndarray 
        (N,M), (N,M), (N,M), (N), (N), (N)
        
        S1 and S2 stand for the scattering function for the given particle 
        diameters (column number) and scattering angles (row number). 
        :math:`\\S_1` stands for light that is polarized perpendicular to the 
        propagation plane, and :math:`S_2` for light polarized parallel to the 
        propagation plane. 
        
        P11 Stands for the intensity component of the phase matrix, this is not
        normalized
        
        The scattering, absorption and extinction efficiencies
        are calculated for each diameter
    
    Polarization parameter:
    1 = accounts for light polarized perpendicular to propagation plane
    2 = accounts for light polarized parallel to propagation plane
    None = returns average
    """
    cos_theta   = np.cos(theta)
    n_angles    = len(cos_theta)
    x           = np.pi*particleDiameters/wavelength
    k           = 2*np.pi / wavelength
    # Calculation of Mie parameters
    (a,b) = mie_abcd(refractiveIndex, x)
    
    n_max         = a.shape[1]
    n_diams       = a.shape[0]
   
    # Calculation of the spherical harmonics Pi and Theta
    pi           = np.empty((n_angles,n_diams,n_max))
    tau          = np.empty((n_angles,n_diams,n_max))
    
    header       = np.tile(Header(n_max), (n_diams,1))
    
    for (i, ctheta) in enumerate(cos_theta):
        pigen        = Pi(n_max, ctheta)
        pitemp       = np.fromiter(pigen, dtype = np.float, count = n_max)
        tautemp      = Tau(pitemp, ctheta)
        pi[i,:,:]    = np.tile(pitemp,  (n_diams, 1))
        tau[i,:,:]   = np.tile(tautemp, (n_diams, 1))
    
    S1 = header * (a * pi + b * tau)
    S2 = header * (b * pi + a * tau)

    S1 = np.sum(S1,2).T
    S2 = np.sum(S2,2).T
    
    ab_header = (2*np.arange(1,n_max+1) + 1) 
    Q_sca     = np.einsum("ij,j->ij",(np.abs(a)**2 + np.abs(b)**2), ab_header)
    Q_sca     = (2 / x**2) * np.sum(Q_sca, 1)
    
    Q_ext     = np.einsum("ij,j->ij", np.real(a + b), ab_header)
    Q_ext     = (2 / x**2) * np.sum(Q_ext, 1)
    
    Q_abs     = Q_ext - Q_sca
    
    # Calculation of the normalized phase function
    P11         = (4 * np.pi / k**2) * 0.5*(np.abs(S1)**2 + np.abs(S2)**2)
    
    #
    # Testing of code correctness using the fundamental extinction
    # formula
    #
    
#     for i in range(len(x)):
#         S12_0 = 0.5*(np.real(S1[i][0]) + np.real(S2[i][0])) 
#         print("d: {:.2e} sigma_ext: {:e}, sigma_ext_fund: {:e}, sigma_sca: {:e}".format(
#                 particleDiameters[i],                                       
#                 Q_ext[i]*0.25*np.pi*particleDiameters[i]**2,
#                 (4*np.pi/k**2)*S12_0,
#                 Q_sca[i]*0.25*np.pi*particleDiameters[i]**2))
            
    return (S1, S2, P11, Q_sca, Q_abs, Q_ext)

def mie_characteristics_distribution(
        refractiveIndex, 
        diameters, 
        percentage, 
        wavelength,
        theta           = np.linspace(0,np.pi,501),
        CHUNK           = 200):
    """
    This function calculates the mean differential scattering cross section of 
    a particle size distribution (described by a cumulative probability
    function). 
    Note that this result shows the behavior of the distribution, which is not
    the behavior of any of the particles.
    
    Parameters
    ----------
    refrativeIndex : real / complex
        The ratio between particle and medium refractive index. The real part 
        of the index is the classical (as used in Snell's law) and the complex 
        is related to light absorption
    particleDiameters : numpy.ndarray (N)
        A list of particle diameters to be calculated
    percentage : numpy.ndarray (N)
        The value of the cumulative distribution function (CDF) of the particle
        diameter distribution sampled at the points given in the
        particleDiameter array
    wavelength : real, meters
        The wavelength of the light source
    theta : numpy.ndarray (M), radians
        A list of scattering angles to be calculated. If not given, defaults
        to a 501-element array from 0 to 180deg
         
    Returns
    -------
    (S1, S2, P11, qs, qa, qe) : tuple of numpy.ndarray (M), numpy.ndarray (M), real, real, real
        The scattering funtions for the given distribution. 
        :math:`S_1` stands for light that is polarized perpendicular to the 
        propagation plane, and :math:`S_2` for light polarized parallel to the 
        propagation plane.
        The values of :math:`Q_{sca}, Q_{abs}, Q_{ext}` are also calculated for
        the distribution
        
        
    Raises
    ------
    ValueError
        If the particle diameter which is being calculated forces the creation
        of too big matrices (happens when the Mie parameter 
        (:math:`2\\pi r \\over \\lambda) is higher than approximately 180)
    """
    n_iters = int(len(diameters) / CHUNK)
    if (len(diameters) % CHUNK != 0):
        n_iters += 1
    
    S_1 = np.zeros_like(theta, dtype = np.complex)
    S_2 = np.zeros_like(theta, dtype = np.complex)
    P11 = np.zeros_like(theta)
    sigma_s = 0
    sigma_a = 0
    sigma_e = 0
    
    k = 2* np.pi / wavelength
    
    for i in range(n_iters):
        diams = diameters[i*CHUNK:(i+1)*CHUNK]
        distr = percentage[i*CHUNK:(i+1)*CHUNK]
        s1,s2, p11, Q_sca, Q_abs, Q_ext = mie_characteristics(
                    refractiveIndex   = refractiveIndex,
                    particleDiameters = diams,
                    wavelength        = wavelength, 
                    theta             = theta)
        S_1 += np.einsum("ij,i->j",  s1, distr)
        S_2 += np.einsum("ij,i->j",  s2, distr)
        P11 += np.einsum("ij,i->j", p11, distr)
        
        areas = (np.pi/4)*diams**2
        
        sigma_s += np.sum(Q_sca*areas*distr)
        sigma_a += np.sum(Q_abs*areas*distr)
        sigma_e += np.sum(Q_ext*areas*distr) 
            
    return (S_1, S_2, P11, sigma_s, sigma_a, sigma_e)

def s_to_sigma(S, wavelength):
    """ Converts between scattering function, S, to differential scattering cross
    section, :math: `\\sigma`
    
    """
    return (np.abs(S)**2) * (wavelength/(2*np.pi))**2

  

  
if __name__ == "__main__":
#     import Utils
#     tic = Utils.Tictoc()
    import time
    import matplotlib.pyplot as plt
    
#     tic.tic()
    t = time.time()
    # Define a range of scattering angles. Will go from 0 to 100 deg in 101
    # steps
    theta = np.linspace(0*np.pi/180,180*np.pi/180,1024)
    # Create a range of diameters
    diam = np.linspace(0, 15, 1024)
#     diam = np.array([0,6.2,10.4,14.2,20,27.4,34.8,44.4])
    # Create a cumulative probability density function representing the particle 
    # size distribution. This can be substituted by another function (This was
    # interpolated from a paper) or a list of values (as lont as the diam and
    # the pdf lists have the same lenght)
    pdf  = scipy.special.gammainc(13.9043,10.9078*diam)**0.2079
#     pdf = np.array([0,0.05,0.15,0.35,0.65,0.85,0.95,1])
    # We need in fact not a cumulative one, but a real distribution, so let's
    # derive it
    perc = np.diff(pdf)
    # We were working with microns, but the code needs meters as input
    # Take care - too big particles will make it crash because it will use too
    # much memory
    diam = diam[1:]*1e-6
    # Now we take the scattering cross section of the distribution. Note that
    # we are taking only one line of the vector (the [0] line), as it
    # represents light polarized perpendicular to the propagation plane
    # (worst case)
    scs1 = mie_characteristics_distribution(
                refractiveIndex = 1.45386, 
                diameters       = diam, 
                percentage      = perc, 
                wavelength      = 532e-9,
                theta           = theta)[2]
                                          
#     tic.toc()
    print("Elapsed time: {:.0f}ms".format(1000*(time.time() - t)))
    # I think you can compare directly the values of SCS1. Here a filtering
    # is done to represent a lens (as they have a finite aperture size, so
    # the angle it will "see" is actually a range
    #
    # Just comment out this area
    apertureAngle = 0
    kernel = np.ones(apertureAngle*len(theta)/180)
    kernel = kernel / len(kernel)
    print( "Kernel is %d elements long" % len(kernel))
    
    #
    #
    #
    pdf, cdf = phase_function_to_pdf_cdf(theta, np.reshape(scs1,(1,-1)))
    
    plt.figure(facecolor = [1,1,1])
    plt.subplot(1,2,1)
    plt.plot(theta*180/np.pi,cdf[0],"k")
    plt.grid()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"CDF")
    
    plt.subplot(1,2,2)
    plt.plot(theta*180/np.pi,pdf[0],"k")
    plt.grid()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"PDF")
    
    # This is useful in case you need to check the particle size distribution
    #for s in scs1:
    #    print s
    #for n,s in enumerate(perc):
    #    print diam[n], s    
#     plt.figure(facecolor = [1,1,1])
#     plt.plot(diam,perc)
#     plt.show()
        
    # Plotting the distribution scattering properties
    plt.figure(facecolor = [1,1,1])
    plt.plot(theta*180/np.pi, np.abs(scs1), label = "Weighted phase function")
    
    # This is used to plot the individual contributions (scattering of a single
    # diameter)
    for n in range(2,len(diam),int(len(diam)/5.)):
        print (n, diam[n])
        scs2 = mie_characteristics_distribution(1.45386, 
                          np.array([diam[n]]), 
                          [1], #np.array([perc[n]]), 
                          532e-9,
                          theta)[2]   
        if apertureAngle > 0:
            # filtered
            plt.plot(theta*180/np.pi,np.convolve(np.abs(scs2),kernel,mode="same"), 
                     label = r"$d = {:.1f}\mu m$".format(diam[n]*1e6))
        else:
            # unfiltered
            plt.plot(theta*180/np.pi,np.abs(scs2), 
                     label = r"$d = {:.1f}\mu m$".format(diam[n]*1e6))
    
    # Formatting the plot
    plt.xlabel("Scattering angle")
    plt.ylabel("Scattering cross section (m^2)/(sr)")
    plt.legend(loc=3)
    
    plt.yscale('log')
    plt.grid(True, which="both", axis="both")
    plt.show()
