"""
.. module :: Toolbox
    :platform: Unix, Windows
    :synopsis: Equipment model
    
This module packs models for equipment usually employed in PIV measurements.

    
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
import matplotlib.pyplot as plt 
import Utils
import Primitives
import MieUtils
from scipy.special import erf, gammainc
from scipy.interpolate import RectBivariateSpline, interp1d
import warnings
import gdal
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
    """
    This class describes a sensor. It is responsible for determining the size
    of the camera field of view and also for recording particles.
    
    The particle recording behavior is similar to the one described by Lecordier
    and Westerweel in their `synthetic image generator 
    <http://link.springer.com/chapter/10.1007%2F978-3-642-18795-7_11>`_ .
    
    Some features can be easily implemented such as:
    
    * Quantum efficiency as a function of wavelength (as the recording function
        receives the wavelength as a parameter)
    * Light field measurement (the "virtualData" field can be used to store 
        more data)
        
    But were not implemented until now.
    """
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
        self.dimension              = np.array([0,  0.0090,  0.0122])
#        self.dimension              = np.array([0,  0.024,  0.036])

        
        #                                      # ROW         # COLUMN
        #                                      #0.0089         0.0118
        self.resolution             = np.array([1200,         1600])
        self.pixelSize              = np.array([7.40,         7.40])*1e-6
        self.fillRatio              = np.array([0.75,         0.75])
        self.fullWellCapacity       = 40e3
        self.quantumEfficiency      = 0.5
        self.gain                   = 2.1 #photons / count
        self.bitDepth               = 14
        self.backgroundMeanLevel    = 244
        self.backgroundNoiseStd     = 50
        self.rawData                = None
        self.saturationData         = None
        self.deadPixels             = None
        self.hotPixels              = None
        self.virtualData            = None
        self.color                  = [1,0,0]
        self.clear()
        
   
    def display(self,colormap='jet'):
        """
        This function displays what is currently recorded in the camera sensor.
        """
        plt.figure(facecolor = [1,1,1])
        #imgplot = plt.imshow(self.readSensor()/(-1+2**self.bitDepth))
        imgplot = plt.imshow(self.readSensor())
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
        Initializes sensor with gaussian noise, the distribution parameters are
        given by:
        
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
        return  np.round(s*(-1+2**self.bitDepth))
    
    def save(self, filename):
        """
        Writes the sensor data in a 16-bit TIFF file (which is compatible
        with some mainstream PIV software)
        
        Parameters
        ----------
        filename : string
            The file name and path (including the extension)
        """
        data = self.readSensor().astype(np.uint16)
        dr = gdal.GetDriverByName("GTiff")
        outDs = dr.Create(filename, 
                          int(self.resolution[1]), 
                          int(self.resolution[0]), 
                          1, 
                          gdal.GDT_Int16) 
        outBand = outDs.GetRasterBand(1)
        outBand.WriteArray(data)
        outBand.FlushCache()
        outBand.SetNoDataValue(-99)
        # This seems to be the wonderful way of gdal to close files, sorry
        dr      = None
        outDs   = None
        outBand = None
    
    def parametricToSensor(self, param_coords):
        """
        From the parametric coordinates :math:`\\\overline{U,V}`,
        which range is :math:`[-1..1]`, calculates the sensor coordinates in
        meters, so the algorithm is basically multiplying by the sensor size.
        
        Parameters
        ----------
        param_coords : numpy.ndarray (N,2)
            The parametric coordinates in the range -1..1. 
            
        Returns
        -------
        sensor_coords : numpy.ndarray (N,2)
            The sensor coordinates in meters
        """
        dim_uv  = self.dimension[1:][::-1]/2
        uvlist  = np.einsum("ij,j->ij",param_coords,dim_uv)
        return uvlist
        
    def sensorToParametric(self, sensor_coords):
        """
        Transform sensor coordinates :math:`(U,V)` in meters to parametric 
        coordinates,
        :math:`(\\\overline{U},\\\overline{V})`.

        
        Parameters
        ----------
        param_coords : numpy.ndarray (N,2)
            The parametric coordinates in the range -1..1. 
            
        Returns
        -------
        sensor_coords : numpy.ndarray (N,2)
            The sensor coordinates in meters
        """
        dim_uv  = 2/self.dimension[1:][::-1]
        uvlist  = np.einsum("ij,j->ij",sensor_coords,dim_uv)
        return uvlist
    
    def sensorToPhysical(self, sensor_coords):
        """
        Transform sensor coordinates :math:`(U,V)` in meters to world coordinates

        
        Parameters
        ----------
        param_coords : numpy.ndarray (N,2)
            The parametric coordinates in the range -1..1. 
            
        Returns
        -------
        sensor_coords : numpy.ndarray (N,3)
            The sensor coordinates in meters
        """
        return self.parametricToPhysical(self.sensorToParametric(sensor_coords))
    
    def sensorToPixel(self, coords):
        """
        Transforms (normalized) parametric coordinates into pixel position on the
        sensor.
        
        There is an inversion of the UV columns because of the unfortunate 
        parametric coordinate system that maps:
         
        u -> sensor.z
        v -> sensor.y
        
        Parameters
        ----------
        coords : numpy.ndarray (N,3)
            The position in the sensor in normalized coordinates (range -1..1)
        
        Returns
        -------
        pixels : numpy.ndarray (N,2)
            The fractional position in sensor pixels in the format [row column] 
        
        DOES NOT CHECK IF OUTSIDE SENSOR BOUNDARIES!!!
        """
        return self.parametricToPixel(self.sensorToParametric(coords))
    
    
    def parametricToPixel(self,coordinates):
        """
        Transforms (normalized) parametric coordinates into pixel position on the
        sensor.
        
        There is an inversion of the UV columns because of the unfortunate 
        parametric coordinate system that maps:
         
        u -> sensor.z
        v -> sensor.y
        
        Parameters
        ----------
        coords : numpy.ndarray (N,3)
            The position in the sensor in normalized coordinates (range -1..1)
        
        Returns
        -------
        pixels : numpy.ndarray (N,2)
            The fractional position in sensor pixels in the format [row column] 
        
        DOES NOT CHECK IF OUTSIDE SENSOR BOUNDARIES!!!
        """
        return 0.5*(coordinates[:,::-1]+1)*self.resolution    
    
    def physicalToPixel(self,coords):   
        """
        Transforms world coordinates into a position in the sensor, given in
        pixels
        
        Parameters
        ----------
        coords : numpy.ndarray (N,3)
            The position in world coordinates
        
        Returns
        -------
        pixels : numpy.ndarray (N,2)
            The fractional position in sensor pixels in the format [row column] 
        
        DOES NOT CHECK IF OUTSIDE SENSOR BOUNDARIES!!!
        """
        return self.physicalToParametric(coords)*self.resolution
    
    def _recordParticles(self,coords,energy,wavelength,diameter):
        """
        coords     - [u,v] sensor coordinates of the recorded point
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
        of computing pulseEnergy.
        
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
            diameter2    = diameter[killist]
            coords2      = coords[killist]
            totalPhotons = totalPhotons[killist]
            sX           = sX[killist]
            sY           = sY[killist]
        else:
            diameter2    = diameter
            coords2      = coords

        pixels   = self.sensorToPixel(coords2) 

        #
        # Auxiliary variables for vectorization
        #
        npts            = np.size(coords2,0)
        
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
        masksize[masksize > np.min(self.resolution)] = np.min(self.resolution) 

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
                 totalPhotons * self.quantumEfficiency / self.gain)

        for (n, (ax,ay)) in enumerate(anchor):
            self.rawData[ax:ax+masksize[0],ay:ay+masksize[1]] += level[n]

    def recordParticles(self,
                        coords,
                        energy,
                        wavelength,
                        diameter,
                        ignoreLarge = 0.2):
        """
        coords     - [u,v] sensor coordinates of the recorded point
        energy     - J -   energy which is captured by the lenses
        wavelength - m -   illumination wavelength
        diameter   - m -   particle image diameter
        
        This is the front-end of the _recordParticle method, its main input
        is an array of sensor coordinates, representing the particle image
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
        # filter out too large particles
        if ignoreLarge > 0:
            threshold = ignoreLarge*np.max(self.dimension)
            if np.size(diameter) == 1 and diameter > threshold:
                return
            
            smallparticles = diameter < threshold
            
            diameter = diameter[smallparticles]
            coords = coords[smallparticles,:] 
            
            if np.size(wavelength) > 1:
                wavelength = wavelength[smallparticles]
            if np.size(energy) > 1:
                energy = energy[smallparticles]     
                      
                
        print "There are %i particles to be recorded" % np.size(coords,0)
                
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
            print "Recording partials in steps of %i" % stepMax             
            k = 0
            nrec = 0
            
            while (k+1)*stepMax < nparts:
                self._recordParticles(coords     [k*stepMax:(k+1)*stepMax], 
                                      energy     [k*stepMax:(k+1)*stepMax], 
                                      wavelength [k*stepMax:(k+1)*stepMax], 
                                      diameter   [k*stepMax:(k+1)*stepMax])
                nrec += np.size(coords[k*stepMax:(k+1)*stepMax],0)
                k = k + 1

            self._recordParticles(coords     [k*stepMax:], 
                                  energy     [k*stepMax:], 
                                  wavelength [k*stepMax:], 
                                  diameter   [k*stepMax:])
            nrec += np.size(coords[k*stepMax:],0)
            print "Recorded %i particles in %i steps" % (nrec, k+1)
        else:
            print "Recording total"
            self._recordParticles(coords, energy, wavelength, diameter)

class Lens(Primitives.Part, Core.PyvsimDatabasable):
    """
    This class represents an objective lens. The implemented model is a thick,
    pupil-centric lens as described by `Aggarwal and Ahuja 
    <http://link.springer.com/article/10.1023%2FA%3A1016324132583>`_ .
    
    This means that the center of projection is assumed to be at the center
    of the entrance pupil.
    
    This class generates the starting vectors for the ray tracing procedure. One
    caveat is that it is not possible to execute ray tracing from the back of the
    lens (i.e. from the sensor to the lens), as the rays are assumed straight
    between these two points. This can be a problem when modelling stacked lenses,
    that have to be substituted by an equivalent one.
    """
    def __init__(self):
        Primitives.Part.__init__(self)
        Core.PyvsimDatabasable.__init__(self)
        self.name            = 'Lens '+str(self._id)
        self.dbName          = "Lenses"
        self.dbParameters    = ["color", "opacity", "diameter", "length",
                                "flangeFocalDistance", "F", "H_fore_scalar",
                                "H_aft_scalar", "E_scalar", "X_scalar",
                                "_Edim", "_Xdim", "distortionParameters"]
        # Plotting parameters
        self.color                        = [0.2,0.2,0.2]
        self.opacity                      = 0.8
        self.diameter                     = 0.076
        self.length                       = 0.091
        self._createPoints()
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
        self.macro                        = False
        self.aperture                     =  2
        self.distortionParameters         = np.array([0,0,0,0,
                                                      0,0,0,0,
                                                      0,0,0,0])
        #Calculated parameters
        self.focusingOffset               = 0
        self.PinholeFore                  = None
        self.PinholeAft                   = None
        self._calculatePositions()
        
    @property
    def H_fore(self):        return self.x * self.H_fore_scalar + self.origin
    @H_fore.setter
    def H_fore(self, h):     self._H_fore_scalar = h 
    @property
    def H_aft(self):         return self.x * self.H_aft_scalar + self.origin
    @H_aft.setter
    def H_aft(self, h):      self.H_aft_scalar = h      
    @property
    def H_fore_scalar(self): return (self._H_fore_scalar + self.focusingOffset)
    @property        
    def H_aft_scalar(self):  return (self._H_aft_scalar + self.focusingOffset)
    @property        
    def E_scalar(self):      return (self._E_scalar + self.focusingOffset)
    @property        
    def X_scalar(self):      return (self._X_scalar + self.focusingOffset)           
    @property
    def E(self):             return self.x*self.E_scalar + self.origin
    @E.setter
    def E(self, e):          self.E_scalar = e
    @property
    def X(self):             return self.x*self.X_scalar + self.origin
    @X.setter
    def X(self, x):          self.X_scalar = x
                        
    @property
    def focusingDistance(self): return self._focusingDistance
    @focusingDistance.setter
    def focusingDistance(self,distance):
        self._focusingDistance = distance
        self._calculatePositions()
        
    @property
    def Edim(self): return self._Edim / self.aperture
    @Edim.setter
    def Edim(self, entrancePupilDiameter): self._Edim = entrancePupilDiameter
        
    @property
    def Xdim(self): return self._Xdim / self.aperture
    @Xdim.setter
    def Xdim(self, pupilDiameter): self._Xdim = pupilDiameter   
    
    def display(self):
        """
        This method creates a plot showing the position of the lens notable
        planes (the two main planes, the pupils and the focusing offset). This
        is intended for debugging purposes, or better understanding how the 
        lens work.
        """
        plt.figure(facecolor = [1,1,1])
        plt.hold(True)
        plt.axis("equal")
        plt.grid(True, which = "both", axis = "both")
        plt.title("Notable planes position, reference - lens flange")
        plt.plot([-self.diameter/2,
                   self.diameter/2,
                   self.diameter/2,
                  -self.diameter/2,
                  -self.diameter/2],
                 [0,0,self.length,self.length,0],
                 "k", 
                 #label="External contour",
                 linewidth=4)
        plt.plot([0,0],
                 [-0.1*self.length,1.1*self.length],
                 "k-.", 
                 #label="Centerline",
                 linewidth=1)
        plt.plot([-self.diameter/1.5,
                   self.diameter/1.5],
                 [self.H_aft_scalar,self.H_aft_scalar],
                 "b", 
                 label="H'",
                 linewidth=2)
        plt.plot([-self.diameter/1.5,
                   self.diameter/1.5],
                 [self.H_fore_scalar,self.H_fore_scalar],
                 "r", 
                 label="H",
                 linewidth=2)     
        plt.plot([-self.diameter/1.5,
                   self.diameter/1.5],
                 [self.focusingOffset,self.focusingOffset],
                 "k", 
                 label="d'",
                 linewidth=1)  
        plt.plot([-self.diameter/1.5,-self.Edim/2],
                 [self.E_scalar,self.E_scalar],
                 "r", 
                 label="E",
                 linewidth=3)  
        plt.plot([self.diameter/1.5,self.Edim/2],
                 [self.E_scalar,self.E_scalar],
                 "r", 
#                 label="E",
                 linewidth=3)         
        plt.plot([-self.diameter/1.5,-self.Xdim/2],
                 [self.X_scalar,self.X_scalar],
                 "b", 
                 label="X",
                 linewidth=3)  
        plt.plot([self.diameter/1.5,self.Xdim/2],
                 [self.X_scalar,self.X_scalar],
                 "b", 
#                 label="X",
                 linewidth=3)            
        plt.legend()
        plt.show()
             
               
    def _calculatePositions(self):
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
        #             -(foc - F - H) +- sqrt((foc-f-H)**2 + 4*F**2))
        #  dprime =  ----------------------------------------------
        #                                  2
        aux     = self.focusingDistance - self.F - (self._H_fore_scalar + 
                                                    self.flangeFocalDistance)
        delta   = aux**2 - 4*(self.F**2)
        d_line1  = (aux - np.sqrt(delta))/2
        d_line2  = (aux + np.sqrt(delta))/2

        if self.macro:
            d_line =  d_line2
        else:
            d_line =  d_line1
        
#        print "------ FOCUS CALCULATION ------------------------"
#        print "foc          : ", self.focusingDistance
#        print "aux          : ", aux
#        print "F            : ", self.F
#        print "H            : ", (self._H_fore_scalar + 
#                                                    self.flangeFocalDistance)
#        print "sqrt(delta)  : ", np.sqrt(delta)
#        print "d_line       : ", d_line
#        print "X            : ", self.X
#        print "Hprime       : ", self.H_aft
#        print "H            : ", self.H_fore
#        print "E            : ", self.E
        
        #
        # Now, we have to find the pseudo position of the pinhole (which is
        # not H_fore.
        v                       = d_line + self.F
        a                       = self._E_scalar - self._H_fore_scalar
        v_bar                   = v + a - v*a/self.F
        
        
#        print "foc   %.5f \nv     %.5f \na     %.5f \nv_bar %.5f" % (self.focusingDistance, 
#                                                                     v, a, v_bar)
        
        self.focusingOffset     = d_line
        
        self.PinholeAft   = self.origin + self.x * (v_bar - 
                                                    self.flangeFocalDistance)
        self.PinholeFore  = self.E
        
    def clearData(self):
        """
        As the data from the Primitives.Part class that has to be cleaned is 
        used only for ray tracing, the parent method is not called.
        
        The notable points, however, are recalculated.
        """
        self._calculatePositions()
        
    def _createPoints(self):
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
        TODO - Implementation of radial distortion model
        """ 
#        return vectors 
        angler = np.arccos(np.einsum("ij,j->i",vectors,self.x))
        angler = np.reshape(angler,(-1,1))
        angley = np.arccos(np.einsum("ij,j->i",vectors,self.y))-np.pi/2
        angley = np.reshape(angley,(-1,1))
        anglez = np.arccos(np.einsum("ij,j->i",vectors,self.z))-np.pi/2
        anglez = np.reshape(anglez,(-1,1))
        
        raxis  = Utils.normalize(np.cross(self.x,vectors))
        yaxis  = np.einsum("j,ij->ij",self.z,np.ones_like(angley))
        zaxis  = np.einsum("j,ij->ij",self.y,np.ones_like(anglez))
#        print "BLAH blah"
#        print self.distortionParameters[0:4]
#        print 
#        print np.hstack([angler,angler**2,angler**3,angler**4])
#        print np.hstack([angley,angley**2,angley**3,angley**4])
        d_angler = np.einsum("j,ij->i",
                             self.distortionParameters[0:4],
                             np.hstack([angler,angler**2,angler**3,angler**4]))
        d_angley = np.einsum("j,ij->i",
                             self.distortionParameters[4:8],
                             np.hstack([angley,angley**2,angley**3,angley**4]))  
        d_anglez = np.einsum("j,ij->i",
                             self.distortionParameters[8:12],
                             np.hstack([anglez,anglez**2,anglez**3,anglez**4]))   
        vectors = Utils.rotateVector(vectors, d_angler, raxis)
        vectors = Utils.rotateVector(vectors, d_angley, yaxis)
        vectors = Utils.rotateVector(vectors, d_anglez, zaxis)     
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
    """
    This class represents a camera composed of a body (used only for display),
    a sensor and a lens, therefore it is an assembly.
    
    The main functions of this class are driving the sensor and the lens 
    together, so that the user can call more logical functions (such as
    initialize) instead of using a complicated series of internal functions.
    
    The camera creates a mapping of world coordinates into sensor coordinates
    by using direct linear transformations (a pinhole model). Lens imperfections
    and ambient influence can be modeled by using several DLTs (which is 
    controlled by the parameter mappingResolution).
    """ 
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
        self._scheimpflugAngle          = 0
        # Mapping properties
        self.mappingResolution          = [2, 2]
        self.circleOfConfusionDiameter  = 29e-6
        self.referenceWavelength        = 532e-9
        self.mapping                    = None
        self.detmapping                 = None
        self.dmapping                   = None
        self.virtualApertureArea        = None
        self.sensorSamplingCenters      = None
        self.physicalSamplingCenters    = None
        # Create and position subcomponents:
        self._positionComponents()
        
    def clearData(self):
        self.mapping                    = None
        self.detmapping                 = None
        self.dmapping                   = None
        self.virtualApertureArea        = None
        self.rawPupilSolidAngle         = None
        self.sensorSamplingCenters      = None
        self.physicalSamplingCenters    = None
        while len(self._items) > 3:
            self.remove(3)
        
    @property
    def bounds(self):
        """
        This signals the ray tracing implementation that no attempt should be
        made to intersect rays with the camera
        """
        return None
    
    @property
    def scheimpflugAngle(self): return self._scheimpflugAngle
    @scheimpflugAngle.setter
    def scheimpflugAngle(self, value):
        """
        This unfortunate vicious behavior is implemented to make sure that the
        _scheimpflugAngle property is always up-to-date, and if one tries to set
        that a meaningful error message is given.
        
        If python supported real overloading, the problem would be solved easily
        """
        raise NotImplementedError("Please use the setScheimpflugAngle function")
    
    def setScheimpflugAngle(self, angle, axis):
        """
        This is a convenience function to set the Scheimpflug angle as in a 
        well-build adapter (which means that the pivoting is performed through
        the sensor center).
        
        Parameters
        ----------
        angle : float (radians)
            The scheimpflug angle
        axis : numpy.array (3)
            The axis of rotation
        """
        self.lens.alignTo(self.x, 
                          self.y, 
                          None, 
                          self.origin + self.x*self.sensorPosition)
        self.rotate(-angle,     axis, self.origin + self.x*self.sensorPosition)
        self.lens.rotate(angle, axis, self.origin + self.x*self.sensorPosition)
        
    def _positionComponents(self):
        """
        This method is a shortcut to define the initial position of the camera,
        there is a definition of the initial positioning of the sensor and the 
        lens.
        """
        if self.lens is not None:
            self.remove(self.lens)
            self.remove(self.body)
            self.remove(self.sensor)
            
        self.lens           = Lens()
        self.sensor         = Sensor()
        self.body           = Primitives.Volume(self.dimension)
        
        self.body.color     = self.color
        self.body.opacity   = self.opacity
        self.body.translate(-self.x*self.dimension[0])
        
        self.append(self.lens)
        self.append(self.sensor)
        self.append(self.body)
        
        # Adaptation in case lens is not compatible with camera (different
        # flange focal distances)
        self.lens.translate(self.x*
                            (self.lens.flangeFocalDistance + 
                             self.sensorPosition))
        
        # Sensor position adjustment
        self.sensor.translate(self.x*self.sensorPosition)
        
    def mapPoints(self, pts, skipPupilAngle = False):
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
            
        skipPupilAngle : boolean
            Setting this flag to true skip the step of calculating the solid
            angle formed by the given points and the pupils. This is used only
            in the initialization of the camera
            
        Returns
        -------
        uv : numpy.array (N,3)
            The points (in sensor homogeneous coordinates) mapped to the
            sensor
        w  : numpy.array (N)
            The distance from the center of projection, as calculated by the
            DLT matrix
        dudv : numpy.array (N,6)
            The derivatives of the coordinates u,v with respect to x,y,z in the
            following order:
            [du/dx,  du/dy,  du/dz,  dv/dx,  dv/dy,  dv/dz]
        lineOfSight : numpy.array (N,3)
            The line of sight vectors (the direction of the light ray that
            goes from the point to the camera center of projection)
        imdim : numpy.array(N)
            The diameter of the image as generated by a point source (consider
            geometrical size + diffraction-limited size)
        pupilSolidAngle : numpy.array(N)
            The solid angle formed by the entrance pupil and the given points. 
            If flag skipPupilAngle is true, returns None
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
        uvw  = np.einsum('ijk,ik->ij',self.mapping[i,j], 
                            np.hstack([pts,np.ones((npts,1))]))
        w           = uvw[:,2] 
        uv          = np.einsum("ij,i->ij", uvw[:,:2], 1/uvw[:,2])
#        print i,j
#        print self.dmapping[i,j].shape 
#        print result.shape                           
        duvw = np.einsum('ijk,ik->ij',self.dmapping[i,j],uvw)
        duvw = np.einsum("ij,i->ij", duvw, 1/w**2)
        dudx    = duvw[:,:3]
        dvdx    = duvw[:,3:]
#        print dudx
        lineofsight  = np.cross(dudx,dvdx)
                    # cheap norm                   # invert if mirror                              
        lineofsightnorm = (np.sqrt(np.sum(lineofsight*lineofsight,1))*
                           np.sign(self.detmapping[i,j]))
        lineofsight  = -lineofsight / np.tile(lineofsightnorm,(3,1)).T
        
        """Calculate the diameter of the geometric image"""
        pts_sensor  = self.sensor.sensorToPhysical(uv)
        
        HpS         = np.sum((self.lens.H_aft - pts_sensor) * self.lens.x,1)
        HpX         = self.lens.X_scalar - self.lens.H_aft_scalar
        
        pprime      = w + self.lens.E_scalar - self.lens.H_fore_scalar
        p           = (self.lens.F * pprime) / (pprime - self.lens.F)

        imdim       = 0*self.lens.Xdim * (p - HpS) / (p - HpX)
        # Diffraction-limited part 
        imdim      += 2.44*self.referenceWavelength*self.lens.aperture
        # Calculates the solid angle "seen" by the points
        if skipPupilAngle:
            pupilSolidAngle = None
        else:
            pupilSolidAngle = self.virtualApertureArea / w**2
        
#        print "W", np.min(w), np.median(w), np.mean(w), np.max(w)
#        print "imdim", np.min(imdim), np.median(imdim), np.mean(imdim), np.max(imdim)
#        print "F", self.lens.F
#        print "p'", np.min(pprime), np.median(pprime), np.mean(pprime), np.max(pprime)
#        print "p", np.min(p), np.median(p), np.mean(p), np.max(p)
#        print "H'S", HpS
#        print "H'X", HpX 
        
        return (uv, w, duvw, lineofsight, imdim, pupilSolidAngle)    
        
    def _shootRays(self, 
                  sensorParamCoords,
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
            bundle.name    = self.name + "RayTracingBundle"
            bundle.append(initialVectors, 
                          self.lens.PinholeFore, 
                          self.referenceWavelength)
            try:
                self.remove(bundle.name)
            except IndexError:
                pass
            
            self.append(bundle)               

        bundle.maximumRayTrace   = maximumRayTrace
        bundle.stepRayTrace      = np.mean(maximumRayTrace) / 2
        bundle.trace(tracingRule = Primitives.RayBundle.TRACING_FOV,
                            restart     = restart) 
        return bundle
        
    def _calculateMappings(self, target):
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
        bundle = self._shootRays(parametricCoords,
                                maximumRayTrace = self.lens.focusingDistance*2)
#        self += bundle
#        self += target
#        import System
#        System.plot(self)

        # Finds the intersections that are important:
        intersections = (target == bundle.rayIntersections)
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
        
        bundle = None
        
        firstInts = np.reshape(firstInts, 
                               (np.size(U,0),np.size(U,1),GLOBAL_NDIM))
        lastInts  = np.reshape(lastInts, 
                               (np.size(U,0),np.size(U,1),GLOBAL_NDIM))  
#        print UV
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
                # We want the DLT to be from meters to meters:
                uvlist = self.sensor.parametricToSensor(uvlist)
                
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
    
    def initialize(self):
        """
        This function calculates the camera field of view and its mapping
        functions to relate world coordinates to sensor coordinates. This
        should be called whenever the ambient is set up, so the camera can 
        be used for synthetic image generation and displaying.
        """
        vv,vh = self._depthOfField()
        # Make sure rays intersect volume by expanding it a little
        vv.expand(0.005)
        self.parent += vv
        self._calculateMappings(vv)
        self.parent.remove(vv)
        
        # Determines the distance mapped by the DLT for the points, so that
        # the solid angle can be calculated individually for each particle.
        # Only vh is used, assuming that astigmatism is not high
        w = self.mapPoints(vh.points, skipPupilAngle = True)[1]
        self.virtualApertureArea = np.mean(vh.data * w**2)
        
    
    def _depthOfField(self):
        """
        This method calculates the camera field of view and depth of field.
        Two volumes are returned - one for vertical focusing and another for
        horizontal focusing (when the ambient has no refractive elements, both
        will probably be the same).
                
        Returns
        -------
        (VV, VH) : pyvsim.Volume
            Each of the volumes represent the region where a point is imaged
            by the camera as a feature with a diameter no greater than 
            "allowableDiameter". Only in the case of astigmatism, VV and VH
            are not the same, then VV (vertical) is the volume where the point 
            is in focus at the camera.y axis and VH (horizontal) is in focus
            at the camera.z axis 
            One important point is that the field .data of the volumes has the
            solid angle formed by the entrance pupil image as "seen" by the
            particle. This is used in scattering calculations
        """
        
        points_param  = np.array([[-1,-1],
                                  [-1,+1],
                                  [+1,+1],
                                  [+1,-1]])
        points        = self.sensor.parametricToPhysical(points_param)
        
        X             = self.lens.Xdim
        HprimeX       = self.lens.H_aft_scalar - self.lens.X_scalar
        dcoc          = self.circleOfConfusionDiameter
        # The distance between sensor extremities and center of fore main
        # plane, projected at the lens optical axis
        points_proj   = np.sum((self.lens.H_aft - points)*self.lens.x,1)
        
        p_prime_fore  = (X*points_proj + dcoc*HprimeX) / (X + dcoc)
        p_prime_aft   = (X*points_proj - dcoc*HprimeX) / (X - dcoc)
#        p_prime_spot  = points_proj
        
        p_fore = self.lens.F*p_prime_fore / (p_prime_fore - self.lens.F)
        p_aft  = self.lens.F*p_prime_aft  / (p_prime_aft  - self.lens.F)
#        p_spot = self.lens.F*p_prime_spot / (p_prime_spot - self.lens.F)
        
        """ Find the vectors emerging from the lens: """
#        print "=--------------------- DOF CALC --------------------------"
#        print "pts\n", points
#        print "H ", self.lens.H_fore
#        print "H'", self.lens.H_aft
#        print "E ", self.lens.E
#        print "X ", self.lens.X
#        print "d'", self.lens.focusingOffset
#        print "fl", self.lens.origin
#        print "ph_aft ", self.lens.PinholeAft
#        print "ph_fore", self.lens.PinholeFore
#        print "p_fore\n", p_fore
#        print "p_aft\n",  p_aft
#        print "p_spot\n", p_spot

        vecs   = self.lens.rayVector(points)
        vecx   = np.sum(vecs*self.lens.x, 1) # projection at optical axis

        p_fore += self.lens.H_fore_scalar - self.lens.E_scalar 
        p_aft  += self.lens.H_fore_scalar - self.lens.E_scalar  
#        p_spot += self.lens.H_fore_scalar - self.lens.E_scalar  

        # "Elongate" points to adapt to ray tracing
        p_fore = np.einsum("i,ij->ij", p_fore * 1/vecx,vecs) + self.lens.E
        p_aft  = np.einsum("i,ij->ij", p_aft  * 1/vecx,vecs) + self.lens.E
#        p_spot = np.einsum("i,ij->ij", p_spot * 1/vecx,vecs) + self.lens.E

        """ p_fore and p_aft are the points in space limiting the in-focus
        region, were there no obstructions, reflection, etc """
        
        p_fore_horz = np.empty_like(p_fore)
        p_fore_vert = np.empty_like(p_fore)
        p_aft_horz = np.empty_like(p_aft)
        p_aft_vert = np.empty_like(p_aft)
        vv_angles  = np.empty(8)
        vh_angles  = np.empty(8)
        
        for n in range(np.size(p_fore,0)):
            (pts, ang) = self._findFocusingPoint(p_fore[n])
            p_fore_vert[n] = pts[0]
            p_fore_horz[n] = pts[1] 
            vv_angles[n] = ang[0]
            vh_angles[n] = ang[1]
            (pts, ang) = self._findFocusingPoint(p_aft[n])
            p_aft_vert[n] = pts[0]
            p_aft_horz[n] = pts[1]
            vv_angles[4+n] = ang[0]
            vh_angles[4+n] = ang[1]

        
        # Remove duplicates (in case calculation has already been done)        
        try:
            self.remove("In-focus-vertical")
        except IndexError:
            pass
        
        try:
            self.remove("In-focus-horizontal")
        except IndexError:
            pass
                
        volume_vert                 = Primitives.Volume()
        volume_vert.surfaceProperty = volume_vert.TRANSPARENT
        volume_vert.name            = "In-focus-vertical"
        volume_vert.color           = np.array([1,0,0])
        volume_vert.opacity         = 0.25
        volume_vert.points          = np.vstack([p_aft_vert,p_fore_vert])
        volume_vert.data            = vv_angles
        self.append(volume_vert)
        
        volume_horz                 = Primitives.Volume()
        volume_horz.surfaceProperty = volume_horz.TRANSPARENT
        volume_horz.name            = "In-focus-horizontal"
        volume_horz.color           = np.array([1,1,0])#np.array(Utils.metersToRGB(referenceWavelength))
        volume_horz.opacity         = 0.25
        volume_horz.points          = np.vstack([p_aft_horz,p_fore_horz])
        volume_horz.data            = vh_angles
        self.append(volume_horz)        
        
        self.rawPupilSolidAngle = 0.5*(np.mean(vv_angles) + np.mean(vh_angles))
        return (volume_vert, volume_horz)
    
    def _findFocusingPoint(self, 
                           theoreticalPoint,
                           angles = np.array([0, 90]), 
                           tol = 1e-3):
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
        angles : numpy.array (N) [0, 90]
            The angles (with relation to the optical axis) for analysis of 
            astigmatism in the system. The typical configuration performs the
            analysis only on the lens xy and xz planes, respectively.         
        tol : float (1e-3)
            The tolerance in finding the ray intersection. This value is kept
            relatively high because in the case of astigmatism, the Y and Z
            axes of the camera might not be aligned with the astigmatism axes,
            which can cause the intersection not to be perfect. This value
            still produces results comparable to the one found in Zeiss 
            datasheets
            
        Returns
        -------
        pts : numpy.array (N,3)
            Each point is the intersection of the marginal rays casted from
            the intersection of the entrance pupil and a plane at an angle
            determined by the parameter "angle" in the input. 
        angle : numpy.array (N)
            The solid angle formed by the point and the entrance pupil
        """
        pupilPoints   = np.ones((2*len(angles),3))
        pupilPoints   = pupilPoints * self.lens.E
        for n, angle in enumerate(angles):
            angle = angle * np.pi / 180
            pupilPoints[n*2]   = (pupilPoints[n*2] + 
                                  self.lens.z*self.lens.Edim*np.cos(angle)/2 +
                                  self.lens.y*self.lens.Edim*np.sin(angle)/2)
            pupilPoints[n*2+1] = (pupilPoints[n*2 + 1] -
                                  self.lens.z*self.lens.Edim*np.cos(angle)/2 -
                                  self.lens.y*self.lens.Edim*np.sin(angle)/2)

#        print "Entrance pupil\n", pupilPoints
        """ Vectors going to the theoretical point """
        vectors       = Utils.normalize(theoreticalPoint - pupilPoints)
        
#        print "Vectors\n", vectors
#        print "Vecnorms\n", np.sqrt(np.sum(vectors*vectors,1))
        """ Create the bundle for ray tracing """
        rays                 = Primitives.RayBundle()
        n                    = self.append(rays)
        rays.append(vectors, pupilPoints, self.referenceWavelength)
        rays.maximumRayTrace = 1.5 * Utils.norm(theoreticalPoint - self.lens.E)
        rays.stepRayTrace    = rays.maximumRayTrace
        rays.trace()
        self.remove(n-1)
        
        """ Now run the bundle trying to find the intersection """
        steps        =  np.size(rays.rayPaths, 0)
        pts          =  np.empty((len(angles),3))
        pupil_angle  =  np.empty(len(angles))

        for step in range(1,steps):
            p2   = rays.rayPaths[step]
            p1   = rays.rayPaths[step-1]
            v    = p2 - p1
            
            for n in range(len(angles)):
                point = Utils.linesIntersection(v [[2*n,2*n+1]], 
                                                p1[[2*n,2*n+1]]) 

                if (Utils.pointSegmentDistance(p1, p2, point) < tol).all():
                    pts[n] = point
                    planar_angle   = np.arccos(np.dot(v[2*n],v[2*n+1])/
                                               (np.linalg.norm(v[2*n])*
                                                np.linalg.norm(v[2*n+1])))
                    pupil_angle[n] = (np.pi/4)*(planar_angle)**2
#                    print "Angle %3.4f, angle %3.4f, solid angle %1.4e" % (angles[n], 
#                                                                           planar_angle*180/np.pi, 
#                                                                           pupil_angle[n])
        rays = None
        return (pts, pupil_angle)
    
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
        phantomPrototype.clearData()
        phantomPrototype.parent             = None
        phantomPrototype.body.color         = [0.5,0,0]
        phantomPrototype.body.opacity       = 0.2
        phantomPrototype.lens.color         = [0.5,0.5,0.5]
        phantomPrototype.lens.opacity       = 0.2
#        print phantomPrototype.x
#        print phantomPrototype.y
#        print phantomPrototype.z
#        print phantomPrototype.lens.x
#        print phantomPrototype.lens.y
#        print phantomPrototype.lens.z
        phantomPrototype.lens.alignTo(phantomPrototype.x, phantomPrototype.y)
        for item in phantomPrototype:
            item.parent = phantomPrototype

        phantomAssembly                     = Primitives.Assembly()
#        sy                                  = self.sensor.dimension[1]
#        sz                                  = self.sensor.dimension[2]
        # Matrix to go from sensor coordinates to sensor
        # local coordinates
        MT  = np.array([[ 0  ,   0,  1],
                        [ 0  ,   1,  0],
                        [ 1  ,   0,  0]])

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
                # coordinates to sensor coords) to local sensor coordinates
                MTM = np.dot(MT, M[:,:-1])
                [_,Qm] = Utils.KQ(MTM)
                    
                phantom.mapping = M
                phantom.alignTo(Qm[0],-Qm[1],None,
                                phantom.lens.PinholeFore, 1e-3) 
                phantomAssembly.append(phantom)
        
        return phantomAssembly
         
class Seeding(Primitives.Assembly):
    def __init__(self):
        Primitives.Assembly.__init__(self)
        self.name                   = 'Seeding ' + str(self._id)
        self.volume                 = Primitives.Volume()
        self.cloud                  = Primitives.Points()
        self                       += self.volume
        self                       += self.cloud
        
        self.density                = 1e11
        
        self._particles             = None
        self._diameters             = None
        self.volume.color           = [0.9,0.9,0.9]
        self.volume.surfaceProperty = self.volume.TRANSPARENT
        
    def refractiveIndex(self, wavelength = 532e-9):
        """
        Returns the index of refraction of the material given the wavelength
        (or a list of them)
        
        Parameters
        ----------
        wavelength : scalar or numpy.array
            The wavelength of the incoming light given in *meters*
        
        Returns
        -------
        refractiveIndex : same dimension as wavelength
            The index of refraction
        """       
        return 1.45386 #self.material.refractiveIndex(wavelength)
        
    @property
    def bounds(self):
        """
        This signals the ray tracing implementation that no attempt should be
        made to intersect rays with the seeding (it's mapped differently)
        """
        return None
            
    @property
    def points(self): return self.cloud.points
    @points.setter
    def points(self, particles):
        self.cloud.points  = particles
        [xmin, ymin, zmin] = np.min(particles,0)
        [xmax, ymax, zmax] = np.max(particles,0)
        self.volume.points = np.array([[xmin,ymax,zmax],
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
        if np.size(diams) != np.size(self.cloud.points,0):
            raise ValueError("The diameter array must be the same size than"+
                             " the particle position array.")
        self._diameters = diams
            
    def seed(self):
        o  = self.volume.points[0]
        v1 = self.volume.points[1] - self.volume.points[0]
        v2 = self.volume.points[3] - self.volume.points[0]
        v3 = self.volume.points[4] - self.volume.points[0] 
        volume = np.linalg.norm(np.dot(v1,np.cross(v2,v3)))
        npts   = volume * self.density
        pts    = np.random.rand(npts,3)
        pts    = o + (np.einsum("i,j->ij",pts[:,0],v1) +   
                      np.einsum("i,j->ij",pts[:,1],v2) +
                      np.einsum("i,j->ij",pts[:,2],v3))
        diams  = np.random.rand(npts)
        diams  = self._sizeDistribution(diams)
        self.cloud.points    = pts
        self._diameters      = diams
        
    def _sizeDistribution(self, seeds):
        diam = np.linspace(0,4,100)
        pdf  = gammainc(13.9043,10.9078*diam)**0.2079
        interpolator = interp1d(pdf, diam)
        return interpolator(seeds)*1e-6  
    
    def scatteredEnergy(self, lineofsight, lightvector, solidangle, 
                        wavelength = 532e-9, polarization = 0):
        lightintensity = Utils.norm(lightvector) 
#        lightvecnorm   = np.einsum("ij,i->ij", lightvector, 1/lightintensity)
        
        scatterangle = np.arccos(np.sum(lineofsight*lightvector,1)/
                                 lightintensity)
        # Replacing nans by suitable value
        scatterangle[np.isnan(scatterangle)] = 9999
        minangle = np.min(scatterangle)
        scatterangle[scatterangle == 9999] = minangle
#         plt.plot(scatterangle)
#         plt.show()
        
        print "Diameter range ", np.min(self.diameters), np.max(self.diameters)
        if np.min(self.diameters) != np.max(self.diameters):
            diams  = np.linspace(np.min(self.diameters),
                                 np.max(self.diameters),
                                 500)
        else:
            diams = np.array([np.min(self.diameters), 
                              np.max(self.diameters)+1e-6])
            
        print "Angle range ", np.min(scatterangle), np.max(scatterangle)
        plt.plot(scatterangle)
        plt.show()
        if np.min(scatterangle) != np.max(scatterangle):
            angles = np.linspace(np.min(scatterangle),
                                 np.max(scatterangle),
                                 30)
        else:
            angles = np.array([np.min(scatterangle), 
                               np.max(scatterangle)+1e-6])
        
        (s1,s2) = MieUtils.mieScatteringCrossSections(self.refractiveIndex(wavelength), 
                                                      diams, 
                                                      wavelength, 
                                                      angles)
        scs = (s1 * (np.cos(polarization)**2) + 
               s2 * (1 - np.cos(polarization)**2))

        interpolant = RectBivariateSpline(angles,
                                          diams,
                                          scs,
                                          kx = 1, ky = 1)
        
        scs         = interpolant.ev(scatterangle, self.diameters)
        return scs*lightintensity*solidangle
    
class CalibrationPlate(Seeding):
    def __init__(self, sidelength = 0.2):
        Seeding.__init__(self)
        [X,Y] = np.meshgrid(np.linspace(-sidelength/2,sidelength/2,100),
                            np.linspace(-sidelength/2,sidelength/2,100))
        Z     = 0*np.ones(np.size(X))
        self.points    = np.vstack((X.ravel(),Y.ravel(),Z)).T
        self.points    = np.vstack([self.points, 
                                    self.points + np.array([1e-3,1e-3,1e-3])])
#        print X.shape, Y.shape, Z.shape, self.points.shape
        self.diameters = (2.2e-6*np.ones(np.size(self.points[:,0])) + 
                          1e-10*np.random.rand(np.size(self.points[:,0])))
        
        
class Laser(Primitives.Assembly):
    def __init__(self):
        Primitives.Assembly.__init__(self)
        self.name                       = 'Laser '+str(self._id)
        self.transientFields.extend(["profileInterpolator"])
        self.body                       = None
        self.rays                       = None
        self.volume                     = None
        # Plotting properties
        self.color                      = [0.1,0.1,0.1]
        self.opacity                    = 1.000
        self.dimension                  = np.array([1.060, 0.250, 0.270])
        self.wavelength                 = 532e-9
        # Beam properties
        self._profile                   = None
        self.profileInterpolator        = None
        self._pulseEnergy               = 0.500
        self._beamDivergence            = np.array([0.0005, 0.25])
        self._beamDiameter              = 0.01#0.018
        [X,Y] = np.meshgrid(np.arange(-6,7), 
                            np.arange(-6,7))
        self.profile                    = np.exp(-0.15*(X**2+Y**2))
        # Ray tracing characteristics
        self.usefulLength               = np.array([1, 3])
        self.usefulLengthDiscretization = 0.1
        self.safeEnergyDensity                 = 5e-3 #1e-3
        self.safetyTracingRays          = [7,5]
        self.safetyTracingStrategy      = [[7,7],
                                           [15,0.05],
                                           [100,100]]
        self._positionComponents()
        
    @property    
    def profile(self): return self._profile
    @profile.setter
    def profile(self, profileMatrix):
        self._profile = profileMatrix
        self.clearData()
        
    @property
    def pulseEnergy(self): return self._pulseEnergy
    @pulseEnergy.setter
    def pulseEnergy(self, power):
        self._pulseEnergy = power
        self.clearData()
        
    @property
    def beamDivergence(self): return self._beamDivergence
    @beamDivergence.setter
    def beamDivergence(self, div):
        self._beamDivergence = div
        self.clearData()
        
    @property
    def beamDiameter(self): return self._beamDiameter
    @beamDiameter.setter
    def beamDiameter(self, diam):
        self._beamDiameter = diam
        self.clearData()
        
    @property
    def bounds(self):
        """
        The laser participates in the ray tracing procedure only if it lays
        the volumes representing its light beam
        """
        if self.volume is not None:
            return self.volume.bounds
        else:
            return None
        
    def display(self):
        plt.figure(facecolor = [1,1,1])
        plt.axis("equal")
        plt.grid(True, which = "both", axis = "both")
        plt.title("Laser beam profile - J/m^2")
        imgplot = plt.imshow(self.profile)
        imgplot.set_interpolation('none')
        plt.colorbar()  
        plt.show()
        
    def clearData(self):
        energy = np.sum(self._profile) * (self.beamDiameter**2 / 
                                          np.size(self._profile))
#        print "Energy", energy
        multiplier = self.pulseEnergy / energy
#        print "Multiplier", multiplier
        self._profile             = self._profile * multiplier
        self.profileInterpolator  = RectBivariateSpline(
                                    np.linspace(-1,1,np.size(self._profile,1)),
                                    np.linspace(-1,1,np.size(self._profile,1)),
                                    self._profile,
                                    kx = 1,
                                    ky = 1)
        if self.volume is not None:
            self.remove(self.volume)
            self.volume                     = None 
            
        self._positionComponents()
        
        Primitives.Assembly.clearData(self)
        
    def _positionComponents(self):
        """
        TODO
        """
        if self.body is None:
            self.body           = Primitives.Volume(self.dimension)
            self.append(self.body)
            self.body.translate(-self.x*self.dimension[0])
            self.body.color     = self.color
            self.body.opacity   = self.opacity
            
        self.rays           = Primitives.RayBundle()
        self.append(self.rays)
        
        vectors             = np.tile(self.x,(4,1))
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

        self.rays.append(vectors, positions, self.wavelength)


    def trace(self):
        """
        Creates an assembly containing volumes representing the laser 
        propagation. The volumes are created from usefulLength[0] to
        usefulLength[1], which is only a way to calculate less elements
        and reduce calculation costs, not an energy relation.
        
        Important parameters
            - Laser.usefulLength : the useful region of the laser sheet
            - Laser.usefulLengthDiscretization : the length of the discretization 
            volume. Shorter elements provide better interpolation, but make
            computation extremely costly.
        """
        self.rays.maximumRayTrace   = self.usefulLength[0]
        self.rays.stepRayTrace      = self.usefulLength[0]
        self.rays.trace(tracingRule = self.rays.TRACING_FOV)
        
        start = np.size(self.rays.rayPaths,0) - 1
        
        self.rays.maximumRayTrace   = self.usefulLength[1]
        self.rays.stepRayTrace      = self.usefulLengthDiscretization
        self.rays.trace(tracingRule = self.rays.TRACING_FOV, restart= True)
        
        end   = np.size(self.rays.rayPaths,0)
        
        self.volume = Primitives.Assembly()
        self.append(self.volume)
        pts = np.array([[-1,-1],
                        [+1,-1],
                        [+1,+1],
                        [-1,+1]])
        for n in range(start, end-1):
            vol                 = Primitives.Volume(fastInit = True)
            vol.surfaceProperty = vol.TRANSPARENT
            vol.points          = np.vstack([self.rays.rayPaths[n],
                                             self.rays.rayPaths[n+1]])
            S1                  = Utils.quadArea(self.rays.rayPaths[n,0], 
                                                 self.rays.rayPaths[n,1], 
                                                 self.rays.rayPaths[n,2], 
                                                 self.rays.rayPaths[n,3])
            S2                  = Utils.quadArea(self.rays.rayPaths[n+1,0], 
                                                 self.rays.rayPaths[n+1,1], 
                                                 self.rays.rayPaths[n+1,2], 
                                                 self.rays.rayPaths[n+1,3])
            vecs                = Utils.normalize(self.rays.rayPaths[n+1]-
                                                  self.rays.rayPaths[n])
            vol.data = np.hstack((np.vstack((pts,pts)),
                                  np.vstack((vecs,vecs)),
                                  np.vstack((np.ones((4,1))*S1,
                                             np.ones((4,1))*S2))))
            vol.color           = Utils.metersToRGB(self.wavelength)
            vol.opacity         = 0.1
            self.volume += vol
            
    def illuminate(self, pts):
        """
        Given a set of points in space, this method calculates the light 
        intensity (in :math:`J/m^2`) and direction produced by the laser.
        
        Parameters
        ----------
        pts : numpy.array (N,3)
            A number of points in space
        
        Returns
        -------
        intensity : numpy.array (N,3)
            A vector which norm is the light intensity (in :math:`J/m^2`) pointing to
            the direction that the light emanating from the laser is
        """
        result = np.zeros((np.size(pts,0),np.size(self.volume[0].data,1)))
        for vol in self.volume:
            result += vol.interpolate(pts)

        [i,j] = result[:,:2].T
        vecs  = Utils.normalize(result[:,2:5])
        S     = result[:,5] 
        intensity = self.profileInterpolator.ev(i, j) * (self.beamDiameter ** 2/
                                                         S)
        intensity[S == 0] = 0
        return np.einsum("ij,i->ij",vecs, intensity)
            
    def traceReflections(self):
        """
        POC implementation of a calculation of laser safety distances
        """
        # Calculate how many rays will be generated
        n1 = self.safetyTracingRays[0]
        n2 = self.safetyTracingRays[1]
        nrays  = n1*n2 
        
        # Create a grid of positions for the rays to start
        x = np.linspace(-1, +1, n2)
        y = np.linspace(-1, +1, n1)
        [X,Y] = np.meshgrid(x,y)
        points = np.vstack([Y.ravel(), 
                            X.ravel(), 
                            np.zeros(nrays)]).T
                            
        # Calculate where these rays should be in the laser output
        physicalPoints = (np.einsum("i,j->ij",points[:,0],self.y*
                          self.beamDiameter*0.5*(n1-1)/(n1-2)) +
                          np.einsum("i,j->ij",points[:,1],self.z*
                          self.beamDiameter*0.5*(n2-1)/(n2-2))).squeeze()

#        physicalPoints = physicalPoints * (self.beamDiameter * (nside-1) / 
#                                           (2*(nside-2)))
        physicalPoints = physicalPoints + self.origin
        
        # For each ray, calculate their corresponding initial propagation vector
        vectors = Utils.quadInterpolation(points, 
                                          np.array([[-1,-1,0],
                                                    [+1,-1,0],
                                                    [+1,+1,0],
                                                    [-1,+1,0]]), 
                                          self.rays.initialVectors)
        
        vectors = Utils.normalize(vectors)
        
        bundle = Primitives.RayBundle()
        bundle.append(vectors, 
                      physicalPoints, 
                      self.wavelength)
        self.append(bundle)       
         
        restart = False
        tic = Utils.Tictoc()
        
        for length, step in self.safetyTracingStrategy:
            print "Tracing up to length %f with step %f" % (length, step)
            bundle.maximumRayTrace  = length
            bundle.stepRayTrace     = step
            tic.tic()
            bundle.trace(tracingRule = bundle.TRACING_LASER_REFLECTION, 
                         restart = restart)
            restart = True
            tic.toc()
        
#        initial_density =  self.pulseEnergy / (self.beamDiameter**2/
#                                               ((n1-1)*(n2-1)))
#        initial_energy  =  self.pulseEnergy / ((n1-2)*(n2-2))
        energyDensity           = np.zeros((n1,n2))
        initialEnergyDensity    = self.profileInterpolator.ev(points[:,0], 
                                                              points[:,1])
        initialEnergyDensity    = np.reshape(initialEnergyDensity,(n1,n2))
        referenceArea           = self.beamDiameter**2 / ((n1-2)*(n2-2))
        maxEnergyDensity        = np.max(initialEnergyDensity)
#        print "maxEd %s" % maxEnergyDensity

        # We will create a connectivity map for a rectangle with side n-1, as
        # we will take the centerpoint for each 4 rays
        n = np.arange((n1-1)*(n2-1))
        n = np.reshape(n,(n1-1,n2-1))

        cts = np.empty(((n2-2)*(n1-2),4))
        k = 0

        for i in np.arange(n1-2):
            for j in np.arange(n2-2):
                cts[k] = [n[i,j], n[i,j+1], n[i+1,j+1], n[i+1,j]]
                k += 1

        cts = cts.astype(int)
                
        color = np.empty((bundle.steps+1,nrays,3))
#        print bundle.steps
#        print color.shape
                

        
        for n in range(np.size(bundle.rayPaths,0)):
            # arrange the rays in the grid they originally were
            pts = np.reshape(bundle.rayPaths[n],(n1,n2,3))
            # find the mean points of each 4 rays
            pts = 0.25*(pts[1:,:-1] + pts[:-1,:-1] + pts[1:,1:] + pts[:-1,1:])
            # reshape in a list of points
            pts = np.reshape(pts,(-1,3))
            # calculate the area of each of the new quads (using our nice
            # connectivity list)
            area = Utils.quadArea(pts[cts[:,0]], 
                                  pts[cts[:,1]],
                                  pts[cts[:,2]],
                                  pts[cts[:,3]])
            area = np.reshape(area,(n1-2,n2-2))
            energyDensity[1:-1,1:-1] = (initialEnergyDensity[1:-1,1:-1] *
                                        referenceArea / area)
            e = energyDensity.ravel()
#            print "E %s" % np.max(e)
            color[n] = Utils.jet(np.log10(e+1e-5),
                                 np.log10(self.safeEnergyDensity),
                                 np.log10(maxEnergyDensity),
                                 saturationIndicator = True)
        
        for n,line in enumerate(bundle):
            line.color = color[:,n]
            line.width = 2
        
        # The rays at the margin (that receive density zero) are then "erased"
        toerase = np.reshape(np.arange(n1*n2),(n1,n2))
        toerase[1:-1,1:-1] = 0 
        toerase = np.nonzero(toerase.ravel())[0] 
        # This trick does not work for
        # ray[0,0], so:
        bundle[0].opacity = 0                                         
        for i in toerase:
            bundle[i].opacity = 0
            
    
if __name__=='__main__':
    import System
    import copy
    import Library
    tic = Utils.Tictoc()
    
    c                               = Camera()
    c.lens.focusingDistance         = 1#.9695 #0.9725
    c.lens.aperture                 = 2
    c.mappingResolution             = [2,2]
    c.lens.distortionParameters     = np.array([0,0,0,0,
                                                0,0,0,0,
                                                0,0,0,0])
    # Put the sensor at the position [0,0,0] to make verification easier
    c.translate(-c.x*c.sensorPosition)
    
    scheimpflug = 0*-0.75*np.pi/180
    c.setScheimpflugAngle(scheimpflug, c.y)
#    c.rotate(-scheimpflug,     c.y, c.x*c.sensorPosition)
#    c.lens.rotate(scheimpflug, c.y, c.x*c.sensorPosition)

    l                               = Laser()
#    l.beamDivergence                = np.array([0.5e-3, 0.25])
    l.beamDivergence                = np.array([-7e-3, 4.596e-2])
    l.pulseEnergy                   = 5.005# 0.1
    l._positionComponents()
    l.alignTo(-l.x, l.y, -l.z, np.array([0.6,0,0]))
    l.translate(np.array([0,0.5,0]))
    l.usefulLength                  = np.array([0.55, 0.8])
    l.usefulLengthDiscretization    = 0.1
    l.safetyTracingRays             = [10,100]
    l.safetyTracingStrategy         = [[4,.01]]
    
    
    
    v                               = Primitives.Volume()
#    v.rotate(np.pi/9, v.z)
    v.opacity                       = 0.1
    v.dimension                     = np.array([0.3, 0.3, 0.3])
    v.material                      = Library.IdealMaterial()
    v.material.value                = 1.33
    v.surfaceProperty               = v.TRANSPARENT
    v.translate(np.array([0.35,0.5,0])) 
    
    v2                              = Primitives.Volume()
    v2.dimension                    = np.array([0.0001, 0.3, 0.3])
    v2.surfaceProperty              = v2.MIRROR
#    v2.surfaceProperty              = v.TRANSPARENT 
    v2.material                     = Library.IdealMaterial()
    v2.material.value               = 1
    v2.translate(np.array([0.5,0,0]))
    v2.rotate(-np.pi/4,v2.z)

#    seed                            = Seeding()
#    seed.points                     = (np.array([0.5,0.5,0]) + 
#                                       0.02*Primitives.Volume.PARAMETRIC_COORDS)
#    seed.density                    = 1e11 / 800*3
#    seed.seed()
    
    seed = CalibrationPlate()
    seed.translate(np.array([0.5,0.5,0]))
    seed.rotate(1.570796327,np.array([1,0,0]))    


    environment = Primitives.Assembly()
    environment += seed
    environment += c
#    environment += v
    environment += v2
    environment += l

#    Some geometrical transformations to make the problem more interesting
    c.rotate(90*np.pi/180,c.lens.x)    
#    environment.rotate(np.pi/0.1314, c.x)
#    environment.rotate(np.pi/27, c.y)
#    environment.rotate(np.pi/2.1, c.z)

    npts = np.size(seed.points,0)
    
    print "Laser sheet tracing"
    tic.tic()
    l.trace()
    tic.toc() 
    
    print "Laser sheet safety tracing"
    tic.tic()
    l.traceReflections()
    tic.toc()     

    print "\nCamera parameter determination"
    tic.tic()
    c.initialize()
    tic.toc()
    
    print c.virtualApertureArea / (np.pi*(0.05/c.lens.aperture)**2)   
    
    """Calculate the position of each point in the sensor"""
    (uv, w, duvw, lineofsight, imdim, sldangle) = c.mapPoints(seed.points)
    
    """Calculate the incoming light"""
    print "\nIllumination phase"
    tic.tic()
    lightvector = l.illuminate(seed.points)
    tic.toc(np.size(seed.points,0))


    tic.tic()
    energy = seed.scatteredEnergy(lineofsight  = lineofsight, 
                                  lightvector  = lightvector, 
                                  solidangle   = sldangle, 
                                  wavelength   = 532e-9, 
                                  polarization = 0)
    tic.toc(npts)

    tic.tic()
    c.sensor.recordParticles(uv, 
                             energy, 
                             532e-9, 
                             np.abs(imdim))
    tic.toc(npts)
    
    print "\nSaving image"
    tic.tic()
    c.sensor.save("test01.tif")
    tic.toc()
    c.sensor.display("jet")
    
    vc = c.virtualCameras(True)
    environment += vc
#    

    System.plot(environment)