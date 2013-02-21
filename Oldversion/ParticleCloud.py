from __future__ import division
import vtk
import numpy as np
import copy
import Utils
import vec
import Object
import ScatteringFunctions

class ParticleCloud(Object.Object):
    def __init__(self):
        # Properties inherited from class object (renders as box)
        Object.Object.__init__(self)
        self.indexOfRefraction          =  0
        self.indexOfRefractionAmbient   =  0
        self.indexOfReflection          = -1
        self.isLightSource              =  0
        self.points                     = np.array([])
        self.connectivity               = np.array([[5,7,4],[5,6,7], # normal +z
                                                   [3,2,1],[0,3,1], # normal -z
                                                   [3,6,2],[6,3,7], # normal +y
                                                   [1,5,4],[4,0,1], # normal -y
                                                   [5,1,6],[1,2,6], # normal +x
                                                   [7,0,4],[7,3,0]]) # normal -x
        self.normals                    = None
        self.color                      = [1,1,1]
        self.opacity                    = 0.2
        # Self properties
        self.particles                  = np.array([])
        self.intensities                = np.array([])
        self.particleDiameter           = 2e-6
        self.particleDiameterDeviation  = 0.5e-6
        self.laserWavelength            = 532e-9
        self.particleIndexOfRefraction  = 1.3333 + 1e-4*1j
        self.scatteringFunction         = ScatteringFunctions.arbitraryScattering
        
    def intersectLineWithPolygon(self,p0,p1,tol=1e-7):
        """
        As this class is not to be rendered with any raytracing, this method is
        overriden and returns None automatically in order to save processing time
        """
        return None
        
    def setBounds(self,bounds):
        """
        Shortcut to defining the seeding box by giving a vector with the following
        values: [xmin,xmax,ymin,ymax,zmin,zmax]
                 0    1    2    3    4    5
         
        """
        pts = [[bounds[0],bounds[2],bounds[4]],
               [bounds[1],bounds[2],bounds[4]],
               [bounds[1],bounds[3],bounds[4]],
               [bounds[0],bounds[3],bounds[4]],
               [bounds[0],bounds[2],bounds[5]],
               [bounds[1],bounds[2],bounds[5]],
               [bounds[1],bounds[3],bounds[5]],
               [bounds[0],bounds[3],bounds[5]]]
        self.points = np.array(pts)
        
    def getIllumination(self,objects):
        for o in objects:
            if o.isLightSource != 0:
                self.laserWavelength    = o.wavelength
                self.intensities        = o.calculateIntensity(self.particles)

    def getDifferentialScatteringCrossSection(self,observationPoint):
        return self.scatteringFunction(self,observationPoint)
                
    def seed(self,density):
        """
        create a particle distribution along the volume
        """
        xyzmin = np.min(self.points,0)
        xyzmax = np.max(self.points,0)
        xyzrange = xyzmax - xyzmin
        volume = 1
        for n in range(3): 
            volume = volume * (xyzmax[n] - xyzmin[n])
        nparticles = density*volume
        self.particles  = np.random.rand(nparticles,3)*xyzrange + xyzmin
        
    def seedUniform(self,nx=25,ny=25,nz=2,particleDiameter=2e-6):
        bounds = self.getBounds()
        X = vec.linspace(bounds[0],bounds[1],nx)
        Y = vec.linspace(bounds[2],bounds[3],ny)
        Z = vec.linspace(bounds[2],bounds[3],nz)
        
        self.particles          = np.zeros((nx*ny*nz,3))
        self.particleDiameter   = particleDiameter
        
        count = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    self.particles[count] = np.array([X[i],Y[j],Z[k]]).T
                    count = count + 1
        
    def seedLena(self):
        self.scatteringFunction         = ScatteringFunctions.arbitraryScattering
        from lena import lena
        bounds = self.getBounds()
        X = vec.linspace(bounds[0],bounds[1],512)
        Y = vec.linspace(bounds[2],bounds[3],512)
        Z = (bounds[4] + bounds[5]) / 2
        
        self.particles          = np.zeros((512*512,3))
        self.particleDiameter   = np.zeros(512*512)
        
        k = 0
        for i in range(512):
            for j in range(512):
                self.particles[k]           = np.array([X[i],Y[j],Z]).T
                self.particleDiameter[k]    = (lena[i,j]) * 1e-6 
                k = k + 1
        # print self.particles
        # print self.particleDiameter
        
if __name__=="__main__":
    """
    Code for unit testing basic functionality of class
    """
    import Utils
    from pprint import pprint
    
    cloud = ParticleCloud()
    cloud.setBounds([0,1,0,1,0,1])
    Utils.displayScenario([cloud])
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    cloud.seed(1e3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cloud.particles[:,0],cloud.particles[:,1],cloud.particles[:,2])
    plt.show()
