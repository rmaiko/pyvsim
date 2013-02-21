from __future__ import division
import numpy as np
import copy
import vtk
import ParticleCloud
import pprint
import time
import threading
import vec
import math
try:
    import scipy as sp
    scipy_flag = True
except:
    scipy_flag = False

def prototypeScatteringFunction(cloud,observationPoint):
    """
    This is the prototype for the functions calculating the light scattered by
    a particle. The function receives:
    
    - cloud        : object of type ParticleCloud with information about the
                     particle and light characteristics
    - observationPoint : [x,y,z] of the observer, if relevant
    
    Must return:
    
    - a vector of scalars (one per particle) with the differential scattering
      cross sections
    """
    raise NotImplementedError

def arbitraryScattering(cloud,observationPoint):
    """
    Very simple scattering model (scattering as a round equivalent mirror) used
    as a placeholder in the ParticleCloud class
    """
    return 0.25 * np.pi * cloud.particleDiameter**2
    
def mieScatteringUniform(cloud,observationPoint):
    """
    Calculates Mie scattering with thew only simplification being assuming constant
    particle diameter
    """
    vo = observationPoint - cloud.particles
    vi = cloud.intensities
    cos_theta = vec.dot(vo,vi) / (vec.norm(vo) * vec.norm(vi))
    diff_energy = []
    for ct in cos_theta:
        diff_energy.append(mieScatteringCrossSection(cloud,ct))
    return np.array(diff_energy)
    
def mieScatteringCrossSection(cloud,cos_theta,polarization=None):
    """
    Auxiliary function in the calculation of Mie scattering, this could not be
    yet optimized, so care should be taken
    
    Polarization parameter:
    1 = accounts for light polarized perpendicular to propagation plane
    2 = accounts for light polarized parallel to propagation plane
    None = returns average
    """
    i1 = 0 # accounts for light polarized perpendicular to propagation plane
    i2 = 0 # accounts for light polarized parallel to propagation plane
    pi_n_1  = 0
    pi_n    = 1
    tau_n   = cos_theta
    
    # Calculation of mie parameters
    Mie_parameters = mie_abcd(cloud.particleIndexOfRefraction, \
                              np.pi*cloud.particleDiameter/cloud.laserWavelength)
    # Calculation of sigma_diff
    for n in range(1,len(Mie_parameters)+1):
        header  = (2*n + 1)/(n*(n+1))
        a_n     = Mie_parameters[n-1,0]
        b_n     = Mie_parameters[n-1,1]
        i1 = i1 + header*(a_n*pi_n  + b_n*tau_n)
        i2 = i2 + header*(a_n*tau_n + b_n*pi_n)

        pi_n_2 = pi_n_1
        pi_n_1 = pi_n
        # This calculates the next values for the functions pi_n and tau_n
        # so that the iteration does not need an if block for each loop
        n_plus1 = n + 1
        pi_n    = ((2*n_plus1-1)/(n_plus1-1))*cos_theta*pi_n_1 - (n_plus1/(n_plus1-1))*pi_n_2;
        tau_n   = n_plus1*cos_theta*pi_n - (n_plus1+1)*pi_n_1;
    
    i1 = np.abs(i1)**2;
    i2 = np.abs(i2)**2;
    #
    # VALIDATION CASE FORMULA:
    #
    # obj.scatteringCrossSection = 2*10000*0.001*(i1)*(1/8)*(lambda/pi).^2;  
    #
    # 2 because (i1+i2)/8 = average
    # 10000 because example is in cm^2
    # 0.001 because example is multiplied by 1e-3 to match charts
    
    #return (i1+i2)*(1/8)*(cloud.laserWavelength/np.pi)**2;  
    if np.isnan(i1+i2):
        return 0
    if polarization is None:
        return (i1+i2)*(1/8)*(cloud.laserWavelength/np.pi)**2
    if polarization == 1:
        return (i1)*(1/8)*(cloud.laserWavelength/np.pi)**2
    if polarization == 2:
        return (i2)*(1/8)*(cloud.laserWavelength/np.pi)**2    
    
def mie_abcd(m, x):
    """
    Computes a matrix of Mie coefficients, a_n, b_n, c_n, d_n, 
    of orders n=1 to nmax, complex refractive index m=m'+im", 
    and size parameter x=k0*a, where k0= wave number 
    in the ambient medium, a=sphere radius; 
    p. 100, 477 in Bohren and Huffman (1983) BEWI:TDD122
    C. Matzler, June 2002
    
    Adapted from Matlab code (Dec 2012)
    """
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

    if scipy_flag:
        bx = sp.special.jv(nu,x)*sqx
        bz = sp.special.jv(nu,z)*sqz
        yx = sp.special.yv(nu,x)*sqx
    else:
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

if __name__=="__main__":
    """
    Code for unit testing basic functionality of class
    The Mie scattering code was validated against "Light Scattering
    Theory" by David W Hahn - University of Florida
    """
    import matplotlib.pyplot as plt
    import ParticleCloud
    import Utils
    
    tic = Utils.Tictoc()
    
    cloud = ParticleCloud.ParticleCloud()

    angles      = np.linspace(0,np.pi,1000)    

    # Mie regime
    cloud.particleIndexOfRefraction = 1.4 
    cloud.particleDiameter          = 1.7e-6
    cloud.laserWavelength           = 532e-9
    
    scs1         = []
    print ""
    print "   ###   TESTING OF MIE SCATTERING CODE    ### "
    print "1000 calculations in the Mie regime"
    tic.reset()
    for theta in angles:
        scs1.append(mieScatteringCrossSection(cloud,np.cos(theta),1))
    tic.toctask(1000)
    scs1 = np.array(scs1)*2*10000*0.001
    
    
    # Mixed regime
    cloud.particleIndexOfRefraction = 1.4 
    cloud.particleDiameter          = 170e-9
    cloud.laserWavelength           = 532e-9
    
    scs2         = []
    
    print "1000 calculations under Mie regime"
    tic.reset()
    for theta in angles:
        scs2.append(mieScatteringCrossSection(cloud,np.cos(theta),1))
    tic.toctask(1000)
    scs2 = np.array(scs2)*2*10000
    
    # Rayleigh regime
    cloud.particleIndexOfRefraction = 1.4 
    cloud.particleDiameter          = 17e-9
    cloud.laserWavelength           = 532e-9
    
    scs3         = []
    
    print "1000 calculations in Rayleigh regime"
    tic.reset()
    for theta in angles:
        scs3.append(mieScatteringCrossSection(cloud,np.cos(theta),1))
    tic.toctask(1000)
    scs3 = np.array(scs3)*2*10000*1e6
    
    ax = plt.subplot(1,1,1)
    l1 = ax.plot(angles*180/np.pi,scs1,'k', label="d = 1.7e-6m * 1e-3")
    l2 = ax.plot(angles*180/np.pi,scs2,'r', label="d = 170e-9m")
    l3 = ax.plot(angles*180/np.pi,scs3,'b', label="d = 17e-9m * 1e6")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,labels)
    
    plt.xlabel("Scattering angle (deg)")
    plt.ylabel("Diff. Scattering Cross Sec (cm^2sr^{-1})")
    plt.title("compare with: 'Light Scattering Theory' by David W Hahn, \n m = 1.4, lambda = 532nm")    
    plt.yscale('log')
    plt.ylim(1e-14,1e-10)
    plt.show()