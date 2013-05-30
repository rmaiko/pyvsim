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
import Utils
import Primitives
from scipy.special import erf
import warnings
import Core

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
        # self.heigth = 0.024                   x      y        z
        self.dimension              = np.array([0,  0.0089,  0.0118])

        
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
        coords = [y,z] (in parametric -1..1 space)
        
        returns:
        [row column] - fractional position in sensor pixels
        
        DOES NOT CHECK IF OUTSIDE SENSOR BOUNDARIES!!!
        """
        return 0.5*(coordinates+1)*self.resolution
    
    def displaySensor(self,colormap='jet'):
        imgplot = plt.imshow(self.readSensor()/(-1+2**self.bitDepth))
        imgplot.set_cmap(colormap)
        imgplot.set_interpolation('none')
        plt.colorbar()
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
        sX           = (diameter/self.pixelSize[0])*(0.44/1.22)
        sY           = (diameter/self.pixelSize[1])*(0.44/1.22)
        #
        # Filtering useless results
        #
        npts         =  np.size(coords,0)
        killist      = (range(1,npts+1)* 
                        (coords[:,0] <= 1.01) * (coords[:,0] >= -1.01) * 
                        (coords[:,1] <= 1.01) * (coords[:,1] >= -1.01) * 
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
        gX0 = erf((X - 0.5 * frX)*2/sX)
        gX1 = erf((X + 0.5 * frX)*2/sX)
        gY0 = erf((Y - 0.5 * frY)*2/sY)
        gY1 = erf((Y + 0.5 * frY)*2/sY)    

        level = (0.5*((gX1-gX0)*(gY1-gY0))* 
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
        if stepMax == 0:
            stepMax = 1
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

class Lens(Primitives.Part, Core.PyvsimDatabasable):
    def __init__(self):
        Primitives.Part.__init__(self)
        Core.PyvsimDatabasable.__init__(self)
        self.name                   = 'Lens '+str(self._id)
        self.dbName             = "Lenses"
        self.dbParameters           = ["color", "opacity", "diameter", "length",
                                       "flangeFocalDistance", "F", "H_fore_scalar",
                                       "H_aft_scalar", "E_scalar", "X_scalar",
                                       "_Edim", "_Xdim", "distortionParameters"]
        # Plotting parameters
        self.color                        = [0.2,0.2,0.2]
        self.opacity                      = 0.8
        self.diameter                     = 0.076
        self.length                       = 0.091
        self.createPoints()
        # Main planes model
        self.flangeFocalDistance          =  0.0440
        self.F                            =  0.1000
        self._H_fore_scalar               =  0.0460
        self._H_aft_scalar                =  0.0560
        self._E_scalar                    =  0.0214
        self._X_scalar                    =  0.0362568218298555
        self._Edim                        =  0.1000
        self._Xdim                        =  0.0802568218
        # Adjustable parameters
        self._focusingDistance            =  10
        self.aperture                     =  2
        self.distortionParameters         = np.array([0,0,0,1])
        #Calculated parameters
        self.focusingOffset               = 0
        self.PinholeFore              = None
        self.PinholeAft                  = None
        self.calculatePositions()
        
    @property
    def H_fore(self):    return self.x * self.H_fore_scalar + self.origin
    @H_fore.setter
    def H_fore(self, h): self._H_fore_scalar = h
    
    @property
    def H_aft(self):    return self.x * self.H_aft_scalar + self.origin
    @H_aft.setter
    def H_aft(self, h): self.H_aft_scalar = h    
        
    @property
    def H_fore_scalar(self): return (self._H_fore_scalar + self.focusingOffset)
    @property        
    def H_aft_scalar(self):  return (self._H_aft_scalar + self.focusingOffset)
    
    @property        
    def E_scalar(self):      return (self._E_scalar + self.focusingOffset)
    @property        
    def X_scalar(self):      return (self._X_scalar + self.focusingOffset)        
          
    @property
    def E(self):    return self.x*self.E_scalar + self.origin
    @E.setter
    def E(self, e): self.E_scalar = e
        
    @property
    def X(self):    return self.x*self.X_scalar + self.origin
    @X.setter
    def X(self, x): self.X_scalar = x
                        
    @property
    def focusingDistance(self): return self._focusingDistance
    @focusingDistance.setter
    def focusingDistance(self,distance):
        self._focusingDistance = distance
        self.calculatePositions()
        
    @property
    def Edim(self): return self._Edim / self.aperture
    @Edim.setter
    def Edim(self, entrancePupilDiameter): self._Edim = entrancePupilDiameter
        
    @property
    def Xdim(self): return self._Xdim / self.aperture
    @Xdim.setter
    def Xdim(self, pupilDiameter): self._Xdim = pupilDiameter        
               
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
        #                                      
        #             -(foc - F - H) - sqrt((foc-f-H)**2 + 4*F**2))
        #  dprime =  ----------------------------------------------
        #                                  2
        aux     = self.focusingDistance - self.F - (self._H_fore_scalar + 
                                                    self.flangeFocalDistance)
        delta   = aux**2 - 4*(self.F**2)
        d_line  = (aux - np.sqrt(delta))/2
        
#        print "------ FOCUS CALCULATION ------------------------"
#        print "foc          : ", self.focusingDistance
#        print "aux          : ", aux
#        print "F            : ", self.F
#        print "H            : ", (self._H_fore_scalar + 
#                                                    self.flangeFocalDistance)
#        print "sqrt(delta)  : ", np.sqrt(delta)
#        print "d_line       : ", d_line
#        print "Hprime       : ", self.H_aft
        
        #
        # Now, we have to find the pseudo position of the pinhole (which is
        # not H_fore.
        v                       = d_line + self.F
        a                       = self._E_scalar - self._H_fore_scalar
        v_bar                   = v + a - v*a/self.F
        
        self.focusingOffset     = d_line
        
        self.PinholeAft   = self.origin + self.x * (v_bar - 
                                                    self.flangeFocalDistance)
        self.PinholeFore  = self.origin + self.x * self.E_scalar
        
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
        return self.lensDistortion(Utils.normalize(self.PinholeAft - p))
    
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
        self.dimension                  = np.array([0.175,0.066,0.084])
        # Geometrical properties
        self.sensorPosition             = -0.017526
        # Mapping properties
        self.mappingResolution          = [2, 2]
        self.mapping                    = None
        self.detmapping                 = None
        self.dmapping                   = None
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
        self.body           = Primitives.Volume(self.dimension)
        
        self.body.color     = self.color
        self.body.opacity   = self.opacity
        self.body.translate(-self.x*self.dimension[0])
        
        self.insert(self.lens)
        self.insert(self.sensor)
        self.insert(self.body)
        
        # Flange focal distance adjustment
        self.sensor.translate(self.x*self.sensorPosition)
        
    def mapPoints(self, pts):
        """
        This method determines the position that a set of points x,y,z map on
        the camera sensor.
        
        In order to optimize calculation speed, a lot of memory is used, but 
        the method seems to run smoothly up to 2M points in a 5x5 mapping (2GB
        of available RAM) 
        
        The number of elements in the bottleneck matrix is:
        
        N = npts * 3 * mappingResolution[0] * mappingResolution[1]
        
        Parameters
        ----------
        pts : numpy.array (N,3)
            A collection of points in the space
            
        Returns
        -------
        uvw : numpy.array (N,3)
            The points (in non-normalized homogeneous coordinates) mapped to the
            sensor
        vector : numpy.array (N,3)
            The line of sight vectors (the direction of the light ray that
            goes from the point to the camera center of projection)
        """
        npts    = np.size(pts,0)
        # Calculate vectors from samplingCenters to given points
        d       = self.physicalSamplingCenters - np.reshape(pts,(npts,1,1,3))
        #print d.shape, d.nbytes
        # Calculate squared norms of vectors
        d       = np.einsum('ijkl,ijkl->ijk',d,d)
        # Flatten arrays to find minima
#        print self.physicalSamplingCenters
#        print d
        d       = np.reshape(d,(npts,-1))
#        print d
        indexes = np.argmin(d,1)
        # Calculate indexes
#        print indexes
        j       = np.mod(indexes,(self.mappingResolution[1]-1))
        i       = ((indexes - j) / (self.mappingResolution[0]-1))
        i       = i.astype('int')
        # Calculate DLT
        result  = np.einsum('ijk,ik->ij',self.mapping[i,j], 
                            np.hstack([pts,np.ones((npts,1))]))
#        print i,j
#        print self.dmapping[i,j].shape 
#        print result.shape                           
        dresult = np.einsum('ijk,ik->ij',self.dmapping[i,j],result)
        dudx    = dresult[:,:3]
        dvdx    = dresult[:,3:]
#        print dudx
        vector  = np.cross(dudx,dvdx)
                    # cheap norm                   # invert if mirror                              
        vecnorm = np.sqrt(np.sum(vector*vector,1))*np.sign(self.detmapping[i,j])
        vector  = -vector / np.tile(vecnorm,(3,1)).T
        return (result, vector)
        
    def shootRays(self, 
                  sensorParamCoords, 
                  referenceWavelength = 532e-9,
                  maximumRayTrace = 10,
                  restart = False):
        """
        This is a convenience method to create a ray bundle departing from the
        fore pinhole (the center of the entrance pupil) to be used in ray 
        tracing.
        
        Parameters
        ----------
        sensorParamCoords : numpy.array (N,2)
            The UV coordinates of the point in the sensor originating the rays
        referenceWavelength : float
            The wavelength (in meters) to be used for casting the rays used in
            determining the camera field of view. When looking for chromatic
            aberrations, more than one mapping (maybe one camera for each color
            component)
        maximumRayTrace : float
            The maximum distance for the ray to be traced
        restart : boolean (False)
            If the previous tracing needs to be continued, setting this flag
            to True will make the process continue from its last point 
            (don't forget to increase maximumRayTrace, which refers to the total
            distance travelled by the ray, including from previous runs)
        """
        if not restart:
            sensorCoords   = self.sensor.parametricToPhysical(sensorParamCoords)
            # Creates vectors to initialize ray tracing for each point in the 
            # sensor 
            initialVectors = self.lens.rayVector(sensorCoords)
            bundle         = Primitives.RayBundle()
            bundle.insert(initialVectors, 
                          self.lens.PinholeFore, 
                          referenceWavelength)
            self.insert(bundle, 3, overwrite = True)

        self.items[3].maximumRayTrace   = maximumRayTrace
        self.items[3].stepRayTrace      = np.mean(maximumRayTrace) / 2
        self.items[3].trace(tracingRule = Primitives.RayBundle.TRACING_FOV,
                            restart     = restart) 
        return self.items[3]
        
    def calculateMapping(self, target, referenceWavelength = 532e-9):
        """
        This method calculates the transformation matrix(ces) to go from
        world coordinates (XYZ) to parametric sensor coordinates (UV). This is
        a method to avoid having to do ray tracing for each particle, when
        generating a synthetic PIV image, for example.
        
        The field of view is mapped using a volume (target), and it is assumed
        that the light path is rectilinear inside it. The field of view is
        discretized in MxN regions, as defined in the mappingResolution 
        property of the camera.
        
        Discretization in more than one volume is only needed in cases where
        the pinhole camera model is not valid, e.g. in the presence of radial
        distortions or refractive elements. In theory any mapping is represented
        in a piecewise linear manner using the domain partition, however this
        makes computation of synthetic images much more expensive.
        
        Parameters
        ----------
        referenceWavelength : float
            The wavelength (in meters) to be used for casting the rays used in
            determining the camera field of view. When looking for chromatic
            aberrations, more than one mapping (maybe one camera for each color
            component)
        """
        # First determine the points in the sensor to be reference for the
        # mapping
        [U,V]  = np.meshgrid(np.linspace(-1,1,self.mappingResolution[1]), 
                             np.linspace(-1,1,self.mappingResolution[0])) 
#        print V
#        print U   
        parametricCoords = np.vstack([U.ravel(), V.ravel()]).T
        UV               = np.reshape(parametricCoords, 
                                      (np.size(U,0),np.size(U,1),2))
#        print UV
        
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
#        print firstInts
#        print lastInts      
        
        XYZ   = (firstInts + lastInts) / 2
        self.sensorSamplingCenters    = (UV[:-1,:-1]  + UV[:-1,1:]  +
                                         UV[1:,1:]  + UV[1:,:-1])/4
        self.physicalSamplingCenters  = (XYZ[:-1,:-1] + XYZ[:-1,1:] +
                                         XYZ[1:,1:]   + XYZ[1:,:-1])/4
                   
        self.mapping    = np.empty((np.size(UV,0)-1,
                                    np.size(UV,1)-1,
                                    3, 4)) #each mapping matrix is 3x4
        self.detmapping = np.empty((np.size(UV,0)-1,
                                    np.size(UV,1)-1)) #determinants are scalar
        self.dmapping   = np.empty((np.size(UV,0)-1,
                                   np.size(UV,1)-1,
                                   6, 3)) #each derivative matrix is 6x3

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
                    (self.mapping[i,j,:,:],
                     self.dmapping[i,j,:,:],
                     self.detmapping[i,j],
                     temp1,
                     _) = Utils.DLT(uvlist,xyzlist)
                except np.linalg.linalg.LinAlgError:
                    self.mapping    = None
                    self.detmapping = None
                    self.dmapping   = None
                    warnings.warn("Could not find a valid mapping", Warning)
                    return
                cond = cond + temp1
        cond = cond / (np.size(self.sensorSamplingCenters)/3)
        return cond
    
    def depthOfField(self,
                     allowableDiameter   = 29e-6,
                     referenceWavelength = 532e-9):
        """
        This method calculates the camera field of view and depth of field.
        Two volumes are returned - one for vertical focusing and another for
        horizontal focusing (when the ambient has no refractive elements, both
        will probably be the same).
        
        Parameters
        ----------
        allowableDiameter : float
            The diameter of the maximum allowable circle of confusion (in 
            meters). The standard value of 29 microns is chosen to match the
            Zeiss lens datasheets
        referenceWavelength : float
            As the volumes are calculated with ray tracing, a wavelength must
            be chosen for the rays that are casted during the procedure. If
            looking for chromatic aberration, the user must repeat the procedure
            for several wavelengths
                
        Returns
        -------
        (VV, VH) : pyvsim.Volume
            Each of the volumes represent the region where a point is imaged
            by the camera as a feature with a diameter no greater than 
            "allowableDiameter". Only in the case of astigmatism, VV and VH
            are not the same, then VV (vertical) is the volume where the point 
            is in focus at the camera.y axis and VH (horizontal) is in focus
            at the camera.z axis 
        """
        
        points_param  = np.array([[-1,-1],
                                  [-1,+1],
                                  [+1,+1],
                                  [+1,-1]])
        points        = self.sensor.parametricToPhysical(points_param)
        
        X             = self.lens.Xdim
        HprimeX       = self.lens.H_aft_scalar - self.lens.X_scalar
        dcoc          = allowableDiameter
        # The distance between sensor extremities and center of fore main
        # plane, projected at the lens optical axis
        points_proj   = np.sum((self.lens.H_aft - points)*self.lens.x,1)
        
        p_prime_fore  = (X*points_proj + dcoc*HprimeX) / (X + dcoc)
        p_prime_aft   = (X*points_proj - dcoc*HprimeX) / (X - dcoc)
        p_prime_spot  = points_proj
        
        p_fore = self.lens.F*p_prime_fore / (p_prime_fore - self.lens.F)
        p_aft  = self.lens.F*p_prime_aft  / (p_prime_aft  - self.lens.F)
        p_spot = self.lens.F*p_prime_spot / (p_prime_spot - self.lens.F)
        
        p_fore = p_fore + self.lens.H_fore_scalar
        p_aft  = p_aft  + self.lens.H_fore_scalar
        p_spot = p_spot + self.lens.H_fore_scalar
        """ Find the vectors emerging from the lens: """
        vecs   = self.lens.rayVector(points)
        vecx   = np.sum(vecs*self.lens.x, 1) # projection at optical axis
        # "Elongate" points to adapt to ray tracing
        p_fore = np.einsum("i,ij->ij",p_fore / vecx, vecs) + self.lens.origin
        p_aft  = np.einsum("i,ij->ij",p_aft  / vecx, vecs) + self.lens.origin
        p_spot = np.einsum("i,ij->ij",p_spot / vecx, vecs) + self.lens.origin
        """ p_fore and p_aft are the points in space limiting the in-focus
        region, were there no obstructions, reflection, etc """
        
        p_fore_horz = np.empty_like(p_fore)
        p_fore_vert = np.empty_like(p_fore)
        p_aft_horz = np.empty_like(p_aft)
        p_aft_vert = np.empty_like(p_aft)
        for n in range(np.size(p_fore,0)):
            (p_fore_vert[n],p_fore_horz[n]) = self._findFocusingPoint(p_fore[n], 
                                                                     referenceWavelength)
            (p_aft_vert[n],p_aft_horz[n]) = self._findFocusingPoint(p_aft[n], 
                                                                   referenceWavelength)
        for n in range(len(self.items)):
            if (self.items[n].name == "In-focus-vertical" or
                self.items[n].name == "In-focus-horizontal"):
                self.remove(n)
        volume_vert                 = Primitives.Volume()
        volume_vert.surfaceProperty = volume_vert.TRANSPARENT
        volume_vert.name            = "In-focus-vertical"
        volume_vert.color           = np.array([1,0,0])#np.array(Utils.metersToRGB(referenceWavelength))
        volume_vert.opacity         = 0.25
        volume_vert.points          = np.vstack([p_aft_vert,p_fore_vert])
        self.insert(volume_vert)
        
        volume_horz                 = Primitives.Volume()
        volume_horz.surfaceProperty = volume_horz.TRANSPARENT
        volume_horz.name            = "In-focus-horizontal"
        volume_horz.color           = np.array([1,1,0])#np.array(Utils.metersToRGB(referenceWavelength))
        volume_horz.opacity         = 0.25
        volume_horz.points          = np.vstack([p_aft_horz,p_fore_horz])
        self.insert(volume_horz)
        
        

#        print "------ DOF CALCULATION ------------------------"
#        print "p'+\n", p_prime_fore
#        print "p'-\n", p_prime_aft
#        print "p+ \n", p_fore
#        print "p- \n", p_aft
#        print "ps \n", p_spot
#        print "dcoc         : ", dcoc
#        print "E diameter   : ", self.lens.Edim
#        print "X diameter   : ", X
#        print "X            : ", self.lens.X_scalar
#        print "HprimeX      : ", HprimeX
#        print "H'           : ", self.lens.H_aft
#        print "points[0]    : ", points[0]
#        print "H'S[0]       : ", points_proj
        
        return (volume_vert, volume_horz)
    
    def _findFocusingPoint(self, theoreticalPoint, wavelength, tol = 1e-3):
        """
        As the environment that the camera is placed can include mirrors
        and refractive materials, the light path has to be calculated with 
        the ray tracing algorithm.
        
        For the given point, four rays are cast - each at a border of the
        entrance pupil. Their initial vector is defined as the one that reaches
        the theoretical point (given).
        
        Then, for each pair (the horizontal and the vertical), the 
        intersection of the ray paths is verified. The intersection point is
        then the point in space where focusing is perfect.
        
        Parameters
        ----------
        theoreticalPoint : numpy.array (3)
            This is the point in space where the camera would be imaging if it
            were isolated (no mirrors, refractions, etc)
        wavelength : float
            As the volumes are calculated with ray tracing, a wavelength must
            be chosen for the rays that are casted during the procedure. If
            looking for chromatic aberration, the user must repeat the procedure
            for several wavelengths
        tol : float (1e-3)
            The tolerance in finding the ray intersection. This value is kept
            relatively high because in the case of astigmatism, the Y and Z
            axes of the camera might not be aligned with the astigmatism axes,
            which can cause the intersection not to be perfect. This value
            still produces results comparable to the one found in Zeiss 
            datasheets
            
        Returns
        -------
        (PV, PH) : tuple of numpy.array (3)
            PV is the point of intersection of the marginal rays casted from
            the Y (camera vertical) extremities of the entrance pupil. PH is
            the point of intersection of the rays casted from the Z (camera
            horizontal) extremities of the entrance pupil.
        """
        pupilPoints   = np.array([self.lens.E + self.lens.y*self.lens.Edim/2,
                                  self.lens.E + self.lens.z*self.lens.Edim/2,
                                  self.lens.E - self.lens.y*self.lens.Edim/2,
                                  self.lens.E - self.lens.z*self.lens.Edim/2])
        # Vectors going to the theoretical point
        vectors       = theoreticalPoint - pupilPoints
        norms         = np.sqrt(np.sum(vectors*vectors,1))
        # Normalize
        vectors       = np.einsum("ij,i->ij",vectors, 1/norms)
        # Create a ray bundle
        rays          = Primitives.RayBundle()
        n = self.insert(rays)
        rays.insert(vectors, pupilPoints, wavelength)
        rays.maximumRayTrace = 10 * np.mean(norms)
        rays.stepRayTrace    = 10 * np.mean(norms)
        rays.trace()
        self.remove(n-1)
        # Now run the bundle trying to find the intersection
        steps =  np.size(rays.rayPaths, 0)
        Ph = None
        Pv = None
        for n in range(1,steps):
            p2   = rays.rayPaths[n]
            p1   = rays.rayPaths[n-1]
            v    = p2 - p1
            pt_vert = Utils.linesIntersection(v[[0,2]], p1[[0,2]])
            pt_horz = Utils.linesIntersection(v[[1,3]], p1[[1,3]])
#            print n
#            print Utils.pointSegmentDistance(p1, p2, pt_vert)
#            print Utils.pointSegmentDistance(p1, p2, pt_horz)
            if (Utils.pointSegmentDistance(p1, p2, pt_vert) < tol).all():
                Pv = pt_vert
            if (Utils.pointSegmentDistance(p1, p2, pt_horz) < tol).all():
                Ph = pt_horz
        return (Pv, Ph)
    
    def virtualCameras(self, centeronly = True):
        """
        Returns an assembly composed of cameras at the position and orientation
        defined by the original camera mapping. E.g. if a camera is looking
        through a mirror, the virtualCamera will be the mirror image of the
        camera.
        
        Parameters
        ----------
        centeronly : boolean
            If the camera mapping resolution has created more than a single 
            mapping matrix (>2), setting this value to True makes the routine
            create only one camera (for the center mapping). Otherwise it will
            create as many cameras as mapping matrices.
            
        Returns
        -------
        virtualCameras : pyvsim.Assembly
            The cameras within this assembly are copies of the original camera
            only with position and orientation changed (and carcass color), so
            they are completely functional.
            Care should be taken, as having too many cameras requires a lot of
            memory (mappings, sensor data is stored in each camera).
        """
        if self.mapping is None:
            raise  ValueError("No mapping available, " +
                              "could not create virtual cameras")
            return None
        phantomPrototype                    = copy.deepcopy(self)
        phantomPrototype.body.color         = [0.5,0,0]
        phantomPrototype.body.opacity       = 0.2
        phantomPrototype.lens.color         = [0.5,0.5,0.5]
        phantomPrototype.lens.opacity       = 0.2
        print phantomPrototype.x
        print phantomPrototype.y
        print phantomPrototype.z
        print phantomPrototype.lens.x
        print phantomPrototype.lens.y
        print phantomPrototype.lens.z
        phantomPrototype.lens.alignTo(phantomPrototype.x, phantomPrototype.y)

        phantomAssembly                     = Primitives.Assembly()
        sy                                  = self.sensor.dimension[1]
        sz                                  = self.sensor.dimension[2]
        # Matrix to go from sensor parametric coordinates to sensor
        # local coordinates
        #   [Sx]    [  0  ,  0   ,  1][u] 
        #   [Sy] =  [  0  , sy/2 ,  0][v] u = Zcamera
        #   [Sz]    [sz/2 ,  0   ,  0][1] v = Ycamera
        MT                                  = np.array([[ 0  ,   0,  1],
                                                        [ 0  ,sy/2,  0],
                                                        [sz/2,   0,  0]])

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
                                  phantom.lens.PinholeFore)

                # Transform the DLT matrix (that originally goes from global
                # coordinates to sensor parametric) to local sensor coordinates
                MTM = np.dot(MT, M[:,:-1])
                [_,Qm] = Utils.KQ(MTM)
                    
                phantom.mapping = M
                phantom.alignTo(Qm[0],-Qm[1],None,
                                phantom.lens.PinholeFore, 1e-3) 
                phantomAssembly.insert(phantom)
        
        return phantomAssembly
    
class Seeding(Primitives.Volume):
    def __init__(self):
        Primitives.Volume.__init__(self)
        self.name           = 'Seeding ' + str(self._id)
        self._particles      = None
        self._diameters      = None
        self.color           = [0.9,0.9,0.9]
        self.opacity         = 0.300        
        self.surfaceProperty = self.TRANSPARENT
        
    @property
    def particles(self): return self._particles
    @particles.setter
    def particles(self, particles):
        self._particles = particles
        [xmin, ymin, zmin] = np.min(particles,0)
        [xmax, ymax, zmax] = np.max(particles,0)
        self.points = np.array([[xmin,ymax,zmax],
                                [xmin,ymin,zmax],
                                [xmin,ymin,zmin],
                                [xmin,ymax,zmin],
                                [xmax,ymax,zmax],
                                [xmax,ymin,zmax],
                                [xmax,ymin,zmin],
                                [xmax,ymax,zmin]])
        
    @property
    def diameters(self): return self._diameters
    @diameters.setter
    def diameters(self, diams):
        if np.size(diams) != np.size(self._particles,0):
            raise ValueError("The diameter array must be the same size than"+
                             " the particle position array.")
         
        
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
            vol          = Primitives.Volume()
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
    import Library
    tic = Utils.Tictoc()
    
    c                               = Camera()
    c.lens.focusingDistance         = 1.5
    c.lens.aperture                 = 22
    c.mappingResolution             = [3, 3]
    c.lens.translate(np.array([0.026474,0,0]))
    c.translate(-c.x*c.sensorPosition)
#    c.lens.rotate(-0.1, c.z)
#    l                               = Laser()
#    l.usefulLength                  = np.array([0.1, 2])
    
    
    v                               = Primitives.Volume()
    v.rotate(np.pi/9, v.z)
    v.opacity                       = 0.1
    v.dimension                     = np.array([0.3, 0.3, 0.3])
    v.material                      = Library.IdealMaterial()
    v.material.value                = 1.333
    v.surfaceProperty               = v.TRANSPARENT
    v.translate(np.array([0.35,0.5,0])) 
    
    v2                              = Primitives.Volume()
    v2.dimension                    = np.array([0.1, 0.3, 0.3])
    v2.surfaceProperty              = v.MIRROR
#    v2.surfaceProperty              = v.TRANSPARENT 
    v2.material                     = Library.IdealMaterial()
    v2.material.value               = 1
    v2.translate(np.array([0.5,0,0]))
    v2.rotate(-np.pi/4,v2.z)

    environment = Primitives.Assembly()
    environment.insert(c)
    environment.insert(v)
    environment.insert(v2)
#    environment.insert(l)
#    l.trace()
    
#    Some geometrical transformations to make the problem more interesting
    c.rotate(np.pi/4,c.x)    
    environment.rotate(np.pi/0.1314, c.x)
    environment.rotate(np.pi/27, c.y)
    environment.rotate(np.pi/2.1, c.z)
    
#    pts = (np.random.rand(400e3,3)-0.5)*0.04 
#    if (v2.surfaceProperty == v2.MIRROR).all():
#        c.calculateMapping(v, 532e-9)
#        pts[:,1] = 0*pts[:,1]
#        pts = pts + np.array([0.5,0.55,0]) 
#        ax = (0,2)        
#    else:
#        c.calculateMapping(v2, 532e-9)
#        pts[:,0] = 0*pts[:,0] 
#        pts = pts + np.array([0.57,0,0])  
#        ax = (1,2)      
            
    vv,vh = c.depthOfField(allowableDiameter = 15e-6)
    print vv.points
    print vh.points
    vv = copy.deepcopy(vv)
    vv.surfaceProperty = vv.TRANSPARENT
    environment.insert(vv)
#    vv.expand(0.01)
    c.calculateMapping(vv, 532e-9)
        
    phantoms = c.virtualCameras()
    
    if phantoms is not None:       
        environment.insert(phantoms)
  
#    tic.tic()
#    (pos,vec) = c.mapPoints(pts)
#    tic.toc(np.size(pts,0))
#    print "Position\n", pos
#    print "Vector\n", vec
    
#    mag = np.sqrt(np.sum(vec[:,ax]*vec[:,ax],1))
#    plt.quiver(-pos[:,0]/pos[:,2],-pos[:,1]/pos[:,2],
#               vec[:,ax[1]],vec[:,ax[0]], mag)

#    plt.quiver(pts[:,ax[0]],pts[:,ax[1]],
#               vec[:,ax[0]],vec[:,ax[1]])
#    plt.axis('equal')
#    plt.grid(True)
#    plt.show()

    System.plot(environment)
#    c.sensor.recordParticles(coords = np.array([[0,0],[0.1,0],[0.05,0.05]]), 
#                             energy = 1e-10, 
#                             wavelength = 532e-9, 
#                             diameter = 0.0001)
#    c.sensor.displaySensor()

#    System.save(environment, "test.dat")
#    
#    ambient = System.load("test.dat")
#
#    System.plot(ambient)