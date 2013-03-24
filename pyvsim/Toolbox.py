"""
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
import matplotlib.pyplot as plt 
import pyvsim.Utils as Utils
import pyvsim.Primitives as Primitives
from scipy.special import erf
import warnings

MEMSIZE     = 1e6
MEM_SAFETY  = 8
STD_PARAM   = 0.040
GLOBAL_TOL  = 1e-8
GLOBAL_NDIM = 3

class Mirror(Primitives.Plane):
    """
    This class initializes a 1x1m mirror, which can be resized by the
    methods::
    
    * :meth:`~Primitives.Plane.dimension` (preferred); or
    * :meth:`~Primitives.Plane.length`
    * :meth:`~Primitives.Plane.heigth`  
    """
    def __init__(self):
        Primitives.Plane.__init__(self)
        self.name               = 'Mirror '+str(self._id)
        self.surfaceProperty    = self.MIRROR
        
class Dump(Primitives.Plane):
    """
    This class initializes a 1x1m beam dump, that will stop any ray that
    arrives at it.
    
    * :meth:`~Primitives.Plane.dimension` (preferred); or
    * :meth:`~Primitives.Plane.length`
    * :meth:`~Primitives.Plane.heigth`  
    """
    def __init__(self):
        Primitives.Plane.__init__(self)
        self.name               = 'Dump '+str(self._id)
        self.surfaceProperty    = self.DUMP

class Sensor(Primitives.Plane):
    def __init__(self):
        """
        --|----------------------------------> 
          |                            |     z
          |                            |
          |                            | 
          |                            |
          |                            |      
          |                            |      
          |                            | 
          |                            |
          v ___________________________| 
           y
        """
        Primitives.Plane.__init__(self)
        self.name                   = 'Sensor '+str(self._id)
        self.length                 = 0.0118 # by definition, the column dimension
        self.heigth                 = 0.0089 # the row dimension
        # self.length = 0.036
        # self.heigth = 0.024
        
        #                                      # ROW         # COLUMN
        #                                      #0.0089         0.0118
        self.resolution             = np.array([1200,         1600])
        self.pixelSize              = np.array([7.40,         7.40])*1e-6
        self.fillRatio              = np.array([0.75,         0.75])
        self.fullWellCapacity       = 40e3
        self.quantumEfficiency      = 0.5
        self.bitDepth               = 14
        self.backgroundMeanLevel    = 10
        self.backgroundNoiseStd     = 10
        self.rawData                = None
        self.saturationData         = None
        self.deadPixels             = None
        self.hotPixels              = None
        self.virtualData            = None
        self.color                  = [1,0,0]
        self.clear()
        
    def parametricToPixel(self,coordinates):
        """
        coords = [y,z] (in parametric 0..1 space)
        
        returns:
        [row column] - fractional position in sensor pixels
        
        DOES NOT CHECK IF OUTSIDE SENSOR BOUNDARIES!!!
        """
        return coordinates*self.resolution
    
    def displaySensor(self,colormap='jet'):
        imgplot = plt.imshow(self.readSensor()/(-1+2**self.bitDepth))
        imgplot.set_cmap(colormap)
        imgplot.set_interpolation('none')
        plt.show()    
    
    def createDeadPixels(self,probability):
        """
        Creates a dead/hot pixel mapping for the sensor reading simulation
        """
        deathMap = np.random.rand(self.resolution[0],self.resolution[1])
        self.deadPixels = deathMap > probability*0.5
        self.hotPixels  = deathMap > 1 - probability*0.5
        
    def clear(self):
        """
        Initializes CCD with gaussian noise, the distribution parameters are
        given by::
        
        * backgroundMeanLevel - the mean value
        * backgroundNoiseVar - the variance of the distribution
        
        To handle negative values, only the absolute value is taken into account
        """
        self.rawData = np.abs(self.backgroundMeanLevel + 
                              self.backgroundNoiseStd * 
                              np.random.randn(self.resolution[0],
                                              self.resolution[1]))
        
    def readSensor(self):
        """
        Returns the sensor reading in counts, creating quantization noise and
        saturating the signal where appropriate. This also simulates dead pixels,
        if there is a dead pixel mapping
        
        The readout noise, however, should be included in the background noise
        property of the class
        """
        s = self.rawData / self.fullWellCapacity
        if self.deadPixels is not None: 
            s = s * self.deadPixels
            s = s + self.hotPixels
        self.saturationData = (s > 1)
        # Corrects if any place is less than zero (may happen due to noise)
        s = s - s*(s < 0)
        # Corrects saturations
        s = s - (s-1)*(s > 1)
        # Return result as counts:
        return np.round(s*(-1+2**self.bitDepth)) 
    
    def physicalToPixel(self,coords):   
        """
        coords = [x,y,z] (in real space)
        
        returns:
        [row column] - fractional position in sensor pixels
        
        DOES NOT CHECK IF OUTSIDE SENSOR BOUNDARIES!!!
        """
        return self.physicalToParametric(coords)*self.resolution
    
    def _recordParticles(self,coords,energy,wavelength,diameter):
        """
        coords     - [y,z] parametric coordinates of the recorded point
        energy     - J -   energy which is captured by the lenses
        wavelength - m -   illumination wavelength
        diameter   - m -   particle image diameter
        
        Writes the diffraction-limited image of a particle to the
        sensor. This logic was described by Lecordier and 
        Westerweel on the SIG publication
        
        The method is vectorized, but that demands the creation of
        very large tensors (X, Y, gX0, gX1, gY0, gY1), which can
        lead to a crash of the python intepreter. This is one of
        the reasons that control of this function is given to the
        public readSensor method.
        
        Another issue is the masksize parameter. This determines the
        dimensions of the tensor, which are::
        
        [np of particles, masksize[0], masksize[1]]
        
        This is because the tensors can be seen as a pile of "stickers" 
        each with a particle image on them. Then they are "glued" to 
        the sensor at the correct position.
        
        The masksize must be big enough that the largest particle image
        will still fit to the "sticker". However, an evaluation of the
        error funcion (which is the integral of the gaussian profile) 
        is executed 4 times for each element of the tensors. So, if we 
        have one very large image and many small ones, we waste a lot
        of computing power.
        
        Do not forget the sign convention for the image
        --|----------------------------------> 
          |                            |     z
          |                            |
          |            A____________   | 
          |            |            |  |
          |            |            |  |      A = anchor
          |            |      C     |  |      C = particle center
          |            |            |  | 
          |            |____________|  |
          v ___________________________| 
           y
           
          When particle image is partially out of the image limits, the 
          computation is done over a partially useless domain, but 
          remains correct.
          
        --|----------------------------------> 
          |                            |     z
          |                            |
          |                            | 
          |               A____________|
          |               |            |      A = anchor
          |               |            |      C = particle center
          |               |            | 
          |               |            |
          v ______________|___________C| 
           y
        
        The program performs an integration of a 2D gaussian distribution 
        over the sensitive areas of the pixel (defined by the fill ratio). 
                     
        """
        # Classical formula  E =    h*c        h = 6.62607e-34 J*s
        #                        ---------     c = 299 792 458 m/s
        #                          lambda
        photonEnergy = 6.62607e-34*299792458/wavelength
        totalPhotons = energy / photonEnergy
        sX = (diameter/self.pixelSize[0])*(0.44/1.22)
        sY = (diameter/self.pixelSize[1])*(0.44/1.22)

        #
        # Filtering useless results
        #
        npts         =   np.size(coords,0)
        killist      = (range(1,npts+1)* 
                        (coords[:,0] <= 1.01) * (coords[:,0] >= -0.01) * 
                        (coords[:,1] <= 1.01) * (coords[:,1] >= -0.01) * 
                        ((totalPhotons / (sX * sY)) > 1))

        killist      = np.nonzero(killist)[0]
        
        if np.size(killist) < 1:
            return
        
        if np.size(killist) != np.size(diameter):
            killist      = killist - 1
            diameter2    = diameter[killist]
            coords2      = coords[killist]
            totalPhotons = totalPhotons[killist]
            sX           = sX[killist]
            sY           = sY[killist]
        else:
            diameter2    = diameter
            coords2      = coords

        pixels   = self.parametricToPixel(coords2) 

        #
        # Auxiliary variables for vectorization
        #
        npts        = np.size(coords2,0)
        
        totalPhotons    = np.reshape(totalPhotons,(npts,1,1))
        pixelX          = np.reshape(pixels[:,0],(npts,1,1))
        pixelY          = np.reshape(pixels[:,1],(npts,1,1)) 
        sX              = np.reshape(sX,(npts,1,1))
        sY              = np.reshape(sY,(npts,1,1))
        frX             = self.fillRatio[0]
        frY             = self.fillRatio[1]

        # this masksize guarantees that the particle image will not be cropped
        # when the particle size is too small
        # 
        # here we take only the worst case

        masksize    = np.zeros(2)
        masksize[0] = np.max(np.round(2.25*diameter2/self.pixelSize[0]))
        masksize[1] = np.max(np.round(2.25*diameter2/self.pixelSize[1])) 
        masksize[masksize < 3] = 3

        # Defines the anchor position (cf. documentation above)
        anchor   = np.round(pixels - masksize/2)
               
        # Important to verify if anchor will not force an incorrect matrix addition
        # Case anchor has negative elements, stick them to the pixel zero 
        anchor[anchor < 0] = 0 * anchor[anchor < 0]
        # Case anchor element is out of the matrix boundaries
        anchor[:,0][anchor[:,0] + masksize[0] > 
                    self.resolution[0]] = (self.resolution - masksize)[0]
        anchor[:,1][anchor[:,1] + masksize[1] > 
                    self.resolution[1]] = (self.resolution - masksize)[1]

        anchorX = anchor[:,0]
        anchorY = anchor[:,1]
        anchorX = np.reshape(anchorX, (npts,1,1))
        anchorY = np.reshape(anchorY, (npts,1,1))
       

        [Y,X]   = np.meshgrid(np.arange(masksize[0]),np.arange(masksize[1]))

        X       = X + anchorX - pixelX 
        Y       = Y + anchorY - pixelY
   
        # Magic Airy integral, in fact this is a 2D integral on the sensitive
        # area of the pixel (defined by the fill ratio)
        gX0 = erf(X - 0.5 * frX)
        gX1 = erf(X + 0.5 * frX)
        gY0 = erf(Y - 0.5 * frY)
        gY1 = erf(Y + 0.5 * frY)    

        level = (0.5*((gX1-gX0)*(gY1-gY0) / (sX * sY))* 
                 totalPhotons*self.quantumEfficiency)

        for (n, (ax,ay)) in enumerate(anchor):
            self.rawData[ax:ax+masksize[0],ay:ay+masksize[1]] += level[n]

    def recordParticles(self,coords,energy,wavelength,diameter):
        """
        coords     - [y,z] parametric coordinates of the recorded point
        energy     - J -   energy which is captured by the lenses
        wavelength - m -   illumination wavelength
        diameter   - m -   particle image diameter
        
        This is the front-end of the _recordParticle method, its main input
        is an array of parametric coordinates, representing the particle image
        centers.
        
        The other inputs can be either arrays (e.g. for particle with 
        varying diameters) or scalars (e.g. for all particles with same
        diameter). It is more or less obvious that the arrays must have
        the same length as the coordinate arrays.
        
        Another issue is that the recording is much less efficient
        when particle image sizes varying size (the source of this issue
        is at the _recordParticles documentation). So some tricks (such
        as sorting by particle size) are used to reduce this effect. This
        procedure is made when the standard deviation exceeds 1/10th of the
        particle diameter (ajustable by the STD_PARAM constant at the
        beginning of the module).
        
        Finally, as the recording itself is vectorized, but uses too much 
        memory, it is made in steps. If you find problems too often, adjust
        the following constants at the header of this module::
        
        MEMSIZE - maximum acceptable number of elements in a numpy.ndarray
        
        MEM_SAFETY - factor of safety
        """
        meanDiameter = np.mean(diameter)
        meanDiameter = meanDiameter     / np.mean(self.pixelSize)
        stepMax      = np.round(MEMSIZE / (MEM_SAFETY * meanDiameter**2))
        nparts       = np.size(coords,0)
        
        if stepMax < nparts and np.std(diameter) > STD_PARAM*np.mean(diameter):
            print "Sorting"
            indexes = np.argsort(diameter)
            coords  = coords[indexes]
        else:
            indexes = None
                 
        # Adjusting the inputs, so that _recordParticle will get uniform
        # arrays always   
        if np.size(energy) > 1 and indexes is not None:
            energy  = energy[indexes]
        else:
            energy  = energy * np.ones(nparts)
            
        if np.size(wavelength) > 1 and indexes is not None:
            wavelength = wavelength[indexes]
        else:
            wavelength = wavelength * np.ones(nparts)
            
        if np.size(diameter) > 1 and indexes is not None:
            diameter = diameter[indexes]   
        else:
            diameter = diameter * np.ones(nparts)
        
        if stepMax < nparts:  
            print "Recording partials"             
            k = 0
            while (k + 1)*stepMax < nparts:
                self._recordParticles(coords     [k*stepMax:(k + 1)*stepMax], 
                                      energy     [k*stepMax:(k + 1)*stepMax], 
                                      wavelength [k*stepMax:(k + 1)*stepMax], 
                                      diameter   [k*stepMax:(k + 1)*stepMax])
                k = k + 1
            self._recordParticles(coords     [k*stepMax:], 
                                  energy     [k*stepMax:], 
                                  wavelength [k*stepMax:], 
                                  diameter   [k*stepMax:])
            print "Recording done in ", k, " steps"
        else:
            print "Recording total"
            self._recordParticles(coords, energy, wavelength, diameter)

class Lens(Primitives.Part):
    def __init__(self):
        Primitives.Part.__init__(self)
        self.name                   = 'Lens '+str(self._id)
        # Plotting parameters
        self.color                      = [0.2,0.2,0.2]
        self.opacity                    = 0.8
        self.diameter                   = 0.076
        self.length                     = 0.091
        self.createPoints()
        # Main planes model
        self.flangeFocalDistance          =  0.0440
        self.F                            =  0.1000
        self.H_scalar                     =  0.0460
        self.H_line_scalar                =  0.0560
        self.E_scalar                     =  0.0214
        self.X_scalar                     =  0.0363
        self._Edim                        =  0.1000
        self._Xdim                        =  0.0802
        # Adjustable parameters
        self._focusingDistance            =  10
        self.aperture                     =  2
        self.distortionParameters         = np.array([0,0,0,1])
        #Calculated parameters
        self.focusingOffset               = 0
        self.PinholeEntrance              = None
        self.PinholeExit                  = None
        self.calculatePositions()
        
    @property
    def H(self):
        return self.x*(self.H_scalar + self.focusingOffset) + self.origin
    @H.setter
    def H(self, h):
        self.H_scalar = h
        
    @property
    def H_line(self):
        return self.x*(self.H_line_scalar + self.focusingOffset) + self.origin
    @H_line.setter
    def H_line(self, h):
        self.H_line_scalar = h
        
    @property
    def E(self):
        return self.x*(self.E_scalar + self.focusingOffset) + self.origin
    @E.setter
    def E(self, e):
        self.E_scalar = e
        
    @property
    def X(self):
        return self.x*(self.X_scalar + self.focusingOffset) + self.origin
    @X.setter
    def X(self, x):
        self.X_scalar = x
                        
    @property
    def focusingDistance(self):
        return self._focusingDistance
    @focusingDistance.setter
    def focusingDistance(self,distance):
        self._focusingDistance = distance
        self.calculatePositions()
        
    @property
    def Edim(self):
        return self._Edim / self.aperture
    @Edim.setter
    def Edim(self, entrancePupilDiameter):
        self._Edim = entrancePupilDiameter
        
    @property
    def Xdim(self):
        return self._Xdim / self.aperture
    @Xdim.setter
    def Xdim(self, entrancePupilDiameter):
        self._Xdim = entrancePupilDiameter        
               
    def calculatePositions(self):
        """
        Calculate some important points, must be called whenever the lens 
        is moved.
        
        The position of the exit pinhole is calculated as proposed by:
        
        Aggarwal, M. & Ahuja, N. A 
        Pupil-centric model of image formation 
        Internation Journal of Computer Vision, 2002, 48, 195-214
        """
        # First, we have to calculate d', which is the distance the lens
        # has to offset to focus at the given "focusingDistance"
        #   1            1                     1
        # -----  =  ----------- + ----------------------------
        #   F          F + d'      focusingDistance - SH - d'
        SH      = self.H_scalar + self.flangeFocalDistance
        c       = self.F * (self.focusingDistance - SH + self.F)
        b       = self.focusingDistance - SH
        d_line  = 0.5*((b - self.F) - np.sqrt((b - self.F)**2 + 
                                              4*(self.F * b - c)))
        #
        # Now, we have to find the pseudo position of the pinhole (which is
        # not H'.
        v                       = d_line + self.F
        a                       = self.E_scalar - self.H_scalar
        v_bar                   = v + a - v*a/self.F
        
        self.focusingOffset = d_line
        
        self.PinholeExit       = self.origin + self.x * (v_bar - 
                                                      self.flangeFocalDistance)
        self.PinholeEntrance       = self.origin + self.x * (self.E_scalar + d_line)
        
    def clearData(self):
        """
        As the data from the Primitives.Part class that has to be cleaned is 
        used only for ray tracing, the parent method is not called.
        
        The notable points, however, are recalculated.
        """
        self.calculatePositions()
        
    def createPoints(self):
        """
        This is needed to plot the lens. This will create and the
        points and the connectivity list of a  tube.
        """
        NPTS    = 20
        pts     = []
        conn    = []
        for l in range(2):
            for n in range(NPTS):
                pts.append(l*self.length*self.x + self.diameter * 0.5 * 
                           ((np.cos(2*np.pi*n/NPTS)*self.y) + 
                            (np.sin(2*np.pi*n/NPTS)*self.z)))
        for n in range(NPTS-1):
            conn.append([n,n+1,n+NPTS])
            conn.append([n+NPTS+1,n+NPTS,n+1])
        conn.append([NPTS-1, 0,        NPTS])
        conn.append([NPTS,   2*NPTS-1, NPTS-1])        
            
        self.points       = np.array(pts) + self.origin
        self.connectivity = np.array(conn)
        
    def rayVector(self,p):
        """
        Given a set of points (e.g. in the sensor), will return a list of
        vectors representing the direction to be followed by ray tracing.       
        """
        return self.lensDistortion(Utils.normalize(self.PinholeExit - p))
    
    def lensDistortion(self, vectors):
        """
        Implementation of radial distortion model
        """ 
        return vectors 
    
#        npts = np.size(vectors,0)
#        # Gets the angle between v and the optical axis
#        Ti = np.arccos(np.sum(self.x * \
#                              np.reshape(vectors,(npts,1,GLOBAL_NDIM)),2)).squeeze()
#                              
#        To = np.sum(self.distortionParameters * \
#                    np.array([Ti**4,Ti**3,Ti**2,Ti]),2)
#        
#        axis = Utils.normalize(np.cross(self.x,vectors))
#        return Utils.rotateVector(vectors,(To-Ti),axis)
            
class Camera(Primitives.Assembly):
    def __init__(self):
        Primitives.Assembly.__init__(self)
        self.name                       = 'Camera '+str(self._id)
        self.lens                       = None
        self.sensor                     = None
        self.body                       = None
        # Plotting properties
        self.color                      = [0.172,0.639,0.937]
        self.opacity                    = 0.650
        self.width                      = 0.084
        self.heigth                     = 0.066
        self.length                     = 0.175
        # Geometrical properties
        self.sensorPosition             = -0.017526
        # Mapping properties
        self.mappingResolution          = [2, 2]
        self.mapping                    = None
        self.sensorSamplingCenters      = None
        self.physicalSamplingCenters    = None
        # Create and position subcomponents:
        self.positionComponents()
        
    def clearData(self):
        self.mapping                    = None
        self.sensorSamplingCenters      = None
        self.physicalSamplingCenters    = None
        while len(self.items) > 3:
            self.remove(3)
        
    @property
    def bounds(self):
        """
        This signals the ray tracing implementation that no attempt should be
        made to intersect rays with the camera
        """
        return None
        
    def positionComponents(self):
        """
        This method is a shortcut to define the initial position of the camera,
        there is a definition of the initial positioning of the sensor and the 
        lens.
        """
        self.lens           = Lens()
        self.sensor         = Sensor()
        self.body           = Primitives.Volume(self.length, self.heigth, self.width)
        
        self.body.color     = self.color
        self.body.opacity   = self.opacity
        self.body.translate(-self.x*self.length)
        
        self.insert(self.lens)
        self.insert(self.sensor)
        self.insert(self.body)
        
        # Flange focal distance adjustment
        self.sensor.translate(self.x*self.sensorPosition)
        
    def shootRays(self, sensorParamCoords, 
                  referenceWavelength = 532e-9,
                  maximumRayTrace = 10,
                  restart = False):
        if not restart:
            sensorCoords   = self.sensor.parametricToPhysical(sensorParamCoords)
            # Creates vectors to initialize ray tracing for each point in the 
            # sensor 
            initialVectors = self.lens.rayVector(sensorCoords)
            bundle         = Primitives.RayBundle()
            bundle.insert(initialVectors, 
                          self.lens.PinholeEntrance, 
                          referenceWavelength)
            self.insert(bundle, 3, overwrite = True)

        self.items[3].maximumRayTrace   = maximumRayTrace
        self.items[3].stepRayTrace      = np.mean(maximumRayTrace) / 2
        self.items[3].trace(tracingRule = Primitives.RayBundle.TRACING_FOV,
                            restart     = restart) 
        return self.items[3]
        
    def calculateMapping(self, target, referenceWavelength = 532e-9):
        # First determine the points in the sensor to be reference for the
        # mapping
        [V,U]  = np.meshgrid(np.linspace(-1,1,self.mappingResolution[1]), 
                             np.linspace(-1,1,self.mappingResolution[0]))    
        parametricCoords = np.vstack([U.ravel(), V.ravel()]).T
        UV               = np.reshape(parametricCoords, 
                                      (np.size(U,0),np.size(U,1),2))
        
        bundle = self.shootRays(parametricCoords, 
                                referenceWavelength,
                                maximumRayTrace = self.lens.focusingDistance*2)
        

        # Finds the intersections that are important:
        intersections = (bundle.rayIntersections == target)
                         
        # Finds first and last points
        intersections   = np.tile(intersections,GLOBAL_NDIM)
        firstInts       = np.zeros_like(bundle.rayPaths[0])
        lastInts        = np.zeros_like(bundle.rayPaths[0])
        mask            = np.ones_like(bundle.rayPaths[0])

        for n in range(np.size(bundle.rayPaths,0)):
            firstInts[mask * intersections[n] == 1] = \
                        bundle.rayPaths[n][mask * intersections[n] == 1]
                        
            mask = (intersections[n] == 0) * (mask == 1)
            
            lastInts[intersections[n] == 1] =  \
                        bundle.rayPaths[n,intersections[n] == 1]
        
        firstInts = np.reshape(firstInts, 
                               (np.size(U,0),np.size(U,1),GLOBAL_NDIM))
        lastInts  = np.reshape(lastInts, 
                               (np.size(U,0),np.size(U,1),GLOBAL_NDIM))  
        #print firstInts
        #print lastInts      
        
        XYZ   = (firstInts + lastInts) / 2
        self.sensorSamplingCenters    = (UV[:-1,:-1]  + UV[:-1,1:]  +
                                         UV[1:,1:]  + UV[1:,:-1])/4
        self.physicalSamplingCenters  = (XYZ[:-1,:-1] + XYZ[:-1,1:] +
                                         XYZ[1:,1:]   + XYZ[1:,:-1])/4
        self.mapping = np.empty((np.size(UV,0)-1,
                                 np.size(UV,1)-1,
                                 3,
                                 GLOBAL_NDIM+1))

        cond = 0
        for i in range(np.size(self.sensorSamplingCenters,0)):
            for j in range(np.size(self.sensorSamplingCenters,1)):
                uvlist  = np.array([UV[i  ,j  ],
                                    UV[i+1,j  ],
                                    UV[i+1,j+1],
                                    UV[i  ,j+1],
                                    UV[i  ,j  ],
                                    UV[i+1,j  ],
                                    UV[i+1,j+1],
                                    UV[i  ,j+1]])
                xyzlist = np.array([firstInts[i  ,j  ],
                                    firstInts[i+1,j  ],
                                    firstInts[i+1,j+1],
                                    firstInts[i  ,j+1],
                                    lastInts[i  ,j  ],
                                    lastInts[i+1,j  ],
                                    lastInts[i+1,j+1],
                                    lastInts[i  ,j+1]])
                try:
                    (self.mapping[i,j,:,:], temp1,_) = Utils.DLT(uvlist,xyzlist)
                except np.linalg.linalg.LinAlgError:
                    self.mapping = None
                    warnings.warn("Could not find a valid mapping", Warning)
                    return

                
                cond = cond + temp1
        cond = cond / (np.size(self.sensorSamplingCenters)/3)
        return cond
    
    def depthOfField(self, referenceWavelength = 532e-9):
        CoC = 29e-6
        points_param  = np.array([[-1,-1],[-1,+1],[+1,+1],[+1,-1]])
        points        = self.sensor.parametricToPhysical(points_param)
        
        # Distance from H' (second main plane) to points in sensor projected
        # at the optical axis
        S_H_line  = np.sum(self.lens.x*(self.lens.H_line - points), 1)
        # Distance from exit pupil (X) to points in sensor, projected again
        S_X       = np.sum(self.lens.x*(self.lens.X - points), 1)
        
        # Distance from H (first main plane) to entrance pinhole
        H_PE      = np.sum(self.lens.x*(self.lens.H - self.lens.PinholeEntrance))
        
        d         = Utils.norm(S_X) * CoC / self.lens.Xdim
        d_near    = 1 / (1/self.lens.F - 1/(S_H_line + d))
        d_far     = 1 / (1/self.lens.F - 1/(S_H_line - d))
        
        # Must compensate for the fact that the pinhole is not the main plane
        d_near    = d_near + H_PE
        d_far     = d_far  + H_PE
        
        # Distance from pinhole exit to points in sensor, no projection
        S_PX      = (self.lens.PinholeExit - 
                     self.sensor.parametricToPhysical(points_param))
        S_PX_x    = np.sum(self.lens.x * S_PX, 1)
        
        # Scaling factors
        VecNear   = S_PX * np.tile(d_near / S_PX_x, (3,1)).T
        VecFar    = S_PX * np.tile(d_far  / S_PX_x, (3,1)).T
        
        bundle    = self.shootRays(points_param, 
                                   referenceWavelength, 
                                   Utils.norm(VecNear))
        Pnear     = bundle.rayPaths[-1]
#        print Utils.norm(VecNear)
        
#        print bundle.rayLength
#        print bundle.rayLastVectors
#        print bundle.rayPaths
        
        bundle    = self.shootRays(None, 
                                   None, 
                                   Utils.norm(VecFar),
                                   restart = True)
#        print Utils.norm(VecFar)
#        print bundle.rayLength
#        print bundle.rayLastVectors
#        print bundle.rayPaths
        Pfar      = bundle.rayPaths[-1]
        
        P         = np.vstack([Pnear, Pfar])
        
        v         = Primitives.Volume()
        v.color   = np.array(Utils.metersToRGB(referenceWavelength))
        v.opacity = 0.25
        v.points  = P
        self.insert(v, 4)
        
        ptsy = np.zeros((4,3))
        ptsz = np.zeros((4,3))
        for n in range(4):
            pupilPoints = np.array([self.lens.E + self.lens.y*self.lens.Edim/2,
                                    self.lens.E + self.lens.z*self.lens.Edim/2,
                                    self.lens.E - self.lens.y*self.lens.Edim/2,
                                    self.lens.E - self.lens.z*self.lens.Edim/2,])
            vectors = (self.lens.PinholeEntrance + 0.5 * (VecNear[n] + VecFar[n]) -
                                                                    pupilPoints)
            bundletest = Primitives.RayBundle()
            bundletest.insert(Utils.normalize(vectors), pupilPoints, 532e-9)
            self.insert(bundletest)
            bundletest.maximumRayTrace = 1.5
            bundletest.stepRayTrace = 1.5
            bundletest.trace()
            p = bundletest.rayPaths[-1]
            v = bundletest.rayPaths[-2] - bundletest.rayPaths[-1]
            ptsy[n] = Utils.linesIntersection(v[[0,2]], p[[0,2]])
            ptsz[n] = Utils.linesIntersection(v[[1,3]], p[[1,3]])
        plane = Primitives.Plane()
        plane.points = ptsy
#        print np.mean(ptsy[:,0])
        plane.color = [1,0,0]
        self.insert(plane)
        plane = Primitives.Plane()
        plane.points = ptsz
        print np.mean(ptsy[:,0]) - np.mean(ptsz[:,0])
        plane.color = [1,1,0]
        self.insert(plane)
        return v

    
    def virtualCameras(self, centeronly = True):
        if self.mapping is None:
            raise  ValueError("No mapping available, " +
                              "could not create virtual cameras")
            return None
        phantomPrototype                    = copy.deepcopy(self)
        phantomPrototype.body.color         = [0.5,0,0]
        phantomPrototype.body.opacity       = 0.2
        phantomPrototype.lens.color         = [0.5,0.5,0.5]
        phantomPrototype.lens.opacity       = 0.2
        phantomPrototype.lens.alignTo(phantomPrototype.x, phantomPrototype.y)

        phantomAssembly                     = Primitives.Assembly()
        sy                                  = self.sensor.dimension[0]
        sz                                  = self.sensor.dimension[1]
        # Matrix to go from sensor parametric coordinates to sensor
        # local coordinates
        MT                                  = np.array([[ 0  ,   0, 1],
                                                        [sy/2,   0, 0],
                                                        [ 0  ,sz/2, 0]])

        if centeronly:
            rangei = [np.round(np.size(self.mapping,0)/2)]
            rangej = [np.round(np.size(self.mapping,1)/2)]
        else:
            rangei = range(np.size(self.mapping,0))
            rangej = range(np.size(self.mapping,1))
            
        for i in rangei:
            for j in rangej:
                phantom = copy.deepcopy(phantomPrototype)
                M = self.mapping[i,j,:,:]
                [_,__,V] = np.linalg.svd(M)                             
                pinholePosition = (V[-1] / V[-1][-1])[:-1]
                phantom.translate(pinholePosition - 
                                  phantom.lens.PinholeEntrance)

                # Transform the DLT matrix (that originally goes from global
                # coordinates to sensor parametric) to local sensor coordinates
                MTM = np.dot(MT, M[:,:-1])
                [_,Qm] = Utils.KQ(MTM)
                    
                phantom.mapping = M
                phantom.alignTo(Qm[0],-Qm[1],None,
                                phantom.lens.PinholeEntrance, 1e-3) 
                phantomAssembly.insert(phantom)
        
        return phantomAssembly
        
class Laser(Primitives.Assembly):
    def __init__(self):
        Primitives.Assembly.__init__(self)
        self.name                       = 'Laser '+str(self._id)
        self.body                       = None
        self.rays                       = None
        self.volume                     = None
        # Plotting properties
        self.color                      = [0.1,0.1,0.1]
        self.opacity                    = 1.000
        self.width                      = 0.250
        self.heigth                     = 0.270
        self.length                     = 1.060
        self.wavelength                 = 532e-9
        # Beam properties
#        self.beamProfile                = self.gaussianProfile
        self.beamDiameter               = 0.009
#        self.beamDivergence             = np.array([0.5e-3, 0.25])
        self.beamDivergence             = np.array([0.0001, 0.25])
        self.power                      = 0.300
        # Ray tracing characteristics
        self.usefulLength               = np.array([1, 3])
        self.safeEnergy                 = 1e-3
        self.safetyTracingResolution    = 20
        self.positionComponents()
        
    @property
    def bounds(self):
        """
        TODO
        """
        if self.volume is not None:
            return self.volume.bounds
        else:
            return None
        
    def positionComponents(self):
        """
        TODO
        """
        self.body           = Primitives.Volume(self.length, self.heigth, self.width)
        self.rays           = Primitives.RayBundle()
        self.insert(self.body)
        self.insert(self.rays)
        
        self.body.translate(-self.x*self.length)
        self.body.color     = self.color
        self.body.opacity   = self.opacity
        
        vectors        = np.tile(self.x,(4,1))
        # Divergence in the xz plane
        vectors = np.vstack([Utils.rotateVector(vectors[:2], 
                                                self.beamDivergence[1]/2,
                                                self.y),
                             Utils.rotateVector(vectors[2:], 
                                                -self.beamDivergence[1]/2,
                                                self.y)])
        # Divergence in the xy plane
        vectors = np.vstack([Utils.rotateVector(vectors[0], 
                                                -self.beamDivergence[0]/2,
                                                self.z),
                             Utils.rotateVector(vectors[[1,2]], 
                                                self.beamDivergence[0]/2,
                                                self.z),
                             Utils.rotateVector(vectors[3], 
                                                -self.beamDivergence[0]/2,
                                                self.z)])
        # Position the four main rays with some spacing, according to the
        # initial beam diameter
        positions   = self.origin + (self.beamDiameter/2 * 
                                     np.array([-self.y -self.z,
                                               +self.y -self.z,
                                               +self.y +self.z,
                                               -self.y +self.z]))

        self.rays.insert(vectors, positions, self.wavelength)


    def trace(self):
        """
        """
        self.rays.maximumRayTrace = self.usefulLength[0]
        self.rays.stepRayTrace    = self.usefulLength[0]
        self.rays.trace(tracingRule = self.rays.TRACING_FOV)
        
        start = np.size(self.rays.rayPaths,0) - 1
        
        self.rays.maximumRayTrace = self.usefulLength[1]
        self.rays.stepRayTrace    = self.usefulLength[1]
        self.rays.trace(tracingRule = self.rays.TRACING_FOV, restart= True)
        
        end   = np.size(self.rays.rayPaths,0)
        
        self.volume = Primitives.Assembly()
        self.insert(self.volume)
        for n in range(start, end-1):
            vol = Primitives.Volume()
            vol.points   = np.vstack([self.rays.rayPaths[n],
                                      self.rays.rayPaths[n+1]])
            vol.color    = Utils.metersToRGB(self.wavelength)
            vol.opacity  = 1
            self.volume.insert(vol)
            
    def traceReflections(self):
        """
        """
        npts = self.safetyTracingResolution**2
        nres = self.safetyTracingResolution
        x = np.linspace(-1, +1, nres)
        [X,Y] = np.meshgrid(x,x)
        points = np.vstack([X.ravel(), 
                            Y.ravel(), 
                            np.zeros(npts)]).T
        physicalPoints = (np.reshape(X.ravel(),(npts,1,1))*self.y +
                          np.reshape(Y.ravel(),(npts,1,1))*self.z).squeeze()
        physicalPoints = physicalPoints * self.beamDiameter / 2
        physicalPoints = physicalPoints + self.origin
        
        pts1 = physicalPoints
        
        vectors = Utils.quadInterpolation(points, 
                                          np.array([[-1,-1,0],
                                                    [+1,-1,0],
                                                    [+1,+1,0],
                                                    [-1,+1,0]]), 
                                          self.rays.initialVectors)
        
        vectors = Utils.normalize(vectors)
        
        bundle = Primitives.RayBundle()
        bundle.insert(vectors, 
                      physicalPoints, 
                      self.wavelength)
        self.insert(bundle)        
        bundle.trace(tracingRule = bundle.TRACING_LASER_REFLECTION,)
        
        energy =  self.power / (nres)**2
#        initdensity = energy / (self.beamDiameter / nres)**2
#        print np.log10(initdensity)
#        print np.log10(self.safeEnergy)
        [I,J] = np.meshgrid(range(nres),range(nres))
        I0 = np.vstack([I[ :-1, :-1].ravel(), J[ :-1, :-1].ravel()]).T
        I1 = np.vstack([I[1:  , :-1].ravel(), J[1:  , :-1].ravel()]).T
        I2 = np.vstack([I[1:  ,1:  ].ravel(), J[1:  ,1:  ].ravel()]).T
        I3 = np.vstack([I[ :-1,1:  ].ravel(), J[ :-1,1:  ].ravel()]).T
#        print I0
        currentEnergy = 10
        volumeCollection = Primitives.Assembly()
        
        while(np.max(currentEnergy) > self.safeEnergy):
#            print "PREPARING TO CONTINUE"
            bundle.maximumRayTrace = bundle.maximumRayTrace + 1000
            bundle.stepRayTrace    = bundle.maximumRayTrace
            pts2 = bundle.rayPaths[-1]
            pts1 = np.reshape(pts1,(nres,nres,3))
            pts2 = np.reshape(pts2,(nres,nres,3))

            currentEnergy = (energy / 
                             np.reshape(Utils.quadArea(pts2[I0[:,0],I0[:,1]],
                                                       pts2[I1[:,0],I1[:,1]],
                                                       pts2[I2[:,0],I2[:,1]],
                                                       pts2[I3[:,0],I3[:,1]]),
                                        (nres-1,nres-1)))

            for i in range(nres-1):
                for j in range(nres-1):
                    if currentEnergy[j,i] > self.safeEnergy:
                        vol        = Primitives.Volume(fastInit = True)
                        vol.points = np.vstack([pts1[i,j],
                                                pts1[i+1,j],
                                                pts1[i+1,j+1],
                                                pts1[i,j+1],
                                                pts2[i,j],
                                                pts2[i+1,j],
                                                pts2[i+1,j+1],
                                                pts2[i,j+1]])
                        vol.color = Utils.jet(currentEnergy[j,i], 
                                              self.safeEnergy, 
                                              self.safeEnergy*100)
                        volumeCollection.insert(vol)
            pts1 = pts2
            bundle.trace(tracingRule = bundle.TRACING_LASER_REFLECTION,
                         restart = True)
            
        self.volume = volumeCollection
        self.insert(volumeCollection)
#        print self.volume.bounds
#        print bundle.steps
        
    
if __name__=='__main__':
    import System
    import copy
#    c                               = Camera()
#    c.lens.focusingDistance         = 0.7
#    c.lens.aperture                 = 4
#    c.mappingResolution             = [2, 2]
#    c.lens.translate(np.array([0.026474,0,0]))
#    c.lens.rotate(-0.1, c.z)
    l                               = Laser()
    
    
    v                               = Primitives.Volume()
    v.opacity                       = 0.1
    v.dimension                     = np.array([0.3, 0.3, 0.3])
    v.refractiveIndexConstant       = 5.666
    v.surfaceProperty               = v.TRANSPARENT
    v.translate(np.array([0.35,0.5,0])) 
    
    v2                              = Primitives.Volume()
    v2.dimension                    = np.array([0.1, 0.3, 0.3])
#    v2.surfaceProperty              = v.MIRROR
    v2.surfaceProperty              = v.MIRROR 
    v2.refractiveIndexConstant      = 3.666
    v2.translate(np.array([0.5,0,0]))
    v2.rotate(-np.pi/4,v2.z)

    environment = Primitives.Assembly()
#    environment.insert(c)
    environment.insert(v)
    environment.insert(v2)
    environment.insert(l)
    l.traceReflections()
    
#    Some geometrical transformations to make the problem more interesting
#    c.rotate(np.pi/4,c.x)    
#    environment.rotate(np.pi/0.1314, c.x)
#    environment.rotate(np.pi/27, c.y)
#    environment.rotate(np.pi/2.1, c.z)
    

#    if (v2.surfaceProperty == v.TRANSPARENT).all():
#        c.calculateMapping(v2, 532e-9)
#    else:
#        c.calculateMapping(v, 532e-9)
#        
#    print c.mapping
#        
#    c.depthOfField()
        
#    phantoms = c.virtualCameras(False)
#    
#    if phantoms is not None:       
#        environment.insert(phantoms)

    System.plot(environment)

#    System.save(environment, "test.dat")
#    
#    ambient = System.load("test.dat")
#
#    System.plot(ambient)