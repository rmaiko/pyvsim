#!/usr/bin/env python
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
import Core
from scipy.special import erf

MEMSIZE     = 1e6
MEM_SAFETY  = 8
STD_PARAM   = 0.040
GLOBAL_TOL  = 1e-6
GLOBAL_NDIM = 3

class Mirror(Core.Plane):
    """
    This class initializes a 1x1m mirror, which can be resized by the
    methods::
    
    * :meth:`~Core.Plane.dimension` (preferred); or
    * :meth:`~Core.Plane.length`
    * :meth:`~Core.Plane.heigth`  
    """
    def __init__(self):
        Core.Plane.__init__(self)
        self.name           = 'Mirror '+str(self._id)
        self.reflectAlways  = True
        
class Dump(Core.Plane):
    """
    This class initializes a 1x1m beam dump, that will stop any ray that
    arrives at it.
    
    * :meth:`~Core.Plane.dimension` (preferred); or
    * :meth:`~Core.Plane.length`
    * :meth:`~Core.Plane.heigth`  
    """
    def __init__(self):
        Core.Plane.__init__(self)
        self.name           = 'Dump '+str(self._id)
        self.terminalAlways = True

class Sensor(Core.Plane):
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
          part2 ___________________________| 
           y
        """
        Core.Plane.__init__(self)
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
        self.rawData = np.abs(self.backgroundMeanLevel + \
                              self.backgroundNoiseStd * \
                              np.random.randn(self.resolution[0],\
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
          part2 ___________________________| 
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
          part2 ______________|___________C| 
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
        killist      = range(1,npts+1)* \
                      (coords[:,0] <= 1.01) * (coords[:,0] >= -0.01) * \
                      (coords[:,1] <= 1.01) * (coords[:,1] >= -0.01) * \
                      ((totalPhotons / (sX * sY)) > 1)

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
        anchor[:,0][anchor[:,0] + masksize[0] > self.resolution[0]] = \
            (self.resolution - masksize)[0]
        anchor[:,1][anchor[:,1] + masksize[1] > self.resolution[1]] = \
            (self.resolution - masksize)[1]

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

        level = 0.5*((gX1-gX0)*(gY1-gY0) / (sX * sY))* \
                totalPhotons*self.quantumEfficiency

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

class Objective(Core.Part):
    def __init__(self):
        Core.Part.__init__(self)
        self.name                   = 'Objective '+str(self._id)
        # Plotting parameters
        self.color                      = [0.2,0.2,0.2]
        self.opacity                    = 0.8
        self.diameter                   = 0.076
        self.length                     = 0.091
        self.createPoints()
        # Main planes model
        self.flangeFocalDistance         =  0.0440
        self.F                           =  0.0350
        self.H                           =  0.0287
        self.H_line                      = -0.0090
        self.E                           =  0.0420
        self.X                           =  0.0124
        self._Edim                       =  0.0350
        self._Xdim                       =  0.0564
        # Adjustable parameters
        self._focusingDistance           =  10
        self.aperture                    =  2
        #Calculated parameters
        self.focusingOffset             = None
        self.PEcenter                   = None
        self.PXcenter                   = None
        self.Ecenter                    = None
        self.Xcenter                    = None
        self.calculatePositions()
              
    @property
    def focusingDistance(self):
        return self._focusingDistance
    @focusingDistance.setter
    def focusingDistance(self,distance):
        self._focusingDistance = distance
        self.calculatePositions()
        
    @property
    def Edim(self):
        return self._Edim * self.aperture
    @Edim.setter
    def Edim(self, entrancePupilDiameter):
        self._Edim = entrancePupilDiameter
        
    @property
    def Xdim(self):
        return self._Xdim* self.aperture
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
        SH      = self.H + self.flangeFocalDistance
        c       = self.F * (self.focusingDistance - SH + self.F)
        b       = self.focusingDistance - SH
        d_line  =     0.5*((b - self.F) - \
                                  np.sqrt((b - self.F)**2 + 4*(self.F * b - c)))
        #
        # Now, we have to find the pseudo position of the pinhole (which is
        # not H'.
        part2                       = d_line + self.F
        a                       = self.E - self.H
        v_bar                   = part2 + a - part2*a/self.F
        
        self.focusingOffset = d_line
        self.PXcenter       = v_bar - self.flangeFocalDistance
        self.PEcenter       = self.origin + self.x * self.E
        self.Ecenter        = self.origin + self.x * self.E
        self.Xcenter        = self.origin + self.x * self.X
        
    def clearData(self):
        """
        As the data from the Core.Part class that has to be cleaned is 
        used only for ray tracing, the parent method is not called.
        
        The notable points, however, are recalculated.
        """
        self.calculatePositions()
        
    def createPoints(self):
        """
        This is needed to plot the objective. This will create and the
        points and the connectivity list of a  tube.
        """
        NPTS    = 20
        pts     = []
        conn    = []
        for l in range(2):
            for n in range(NPTS):
                pts.append(l*self.length*self.x + self.diameter * 0.5 * \
                           ((np.cos(2*np.pi*n/NPTS)*self.y) + \
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
        return self.lensDistortion(Utils.normalize(self.PXcenter - p))
    
    def lensDistortion(self,part2):
        """
        Implementation of radial distortion model
        """ 
        npts = np.size(part2,0)
        # Gets the angle between part2 and the optical axis
        Ti = np.arccos(np.sum(self.x * \
                              np.reshape(part2,(npts,1,GLOBAL_NDIM)),2)).squeeze()
        To = np.sum(self.distortionParameters * \
                    np.array([Ti**4,Ti**3,Ti**2,Ti]),2)
        
        axis = Utils.normalize(np.cross(self.x,part2))
        return Utils.rotateVector(part2,(To-Ti),axis)
            
class Camera(Core.Assembly):
    def __init__(self):
        Core.Assembly.__init__(self)
        self.objective                  = None
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
        # Create and position subcomponents:
        self.positionComponents()
        
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
        objective.
        """
        self.objective      = Objective()
        self.sensor         = Sensor()
        self.body           = Core.Volume(self.length, self.heigth, self.width)
        
        self.body.color     = self.color
        self.body.opacity   = self.opacity
        self.body.translate(-self.x*self.length)
        
        self.insert(self.objective)
        self.insert(self.sensor)
        self.insert(self.body)
        
        # Flange focal distance adjustment
        self.sensor.translate(self.x*self.sensorPosition)
        
            
if __name__=='__main__':
    import System
    environment = Core.Assembly()
    c = Camera()
    c.rotate(np.pi/4,c.y)
    #c.translate(np.array([0.5,0.5,0.5]))
    #p = Sensor()
    
    environment.insert(c)
    #environment.insert(p)
    
    pl = System.Plotter()
    environment.acceptVisitor(pl)
    pl.display()
#    sv = System.JSONSaver()
#    environment.acceptVisitor(sv)
#    sv.dump("test.dat")
    
#    s = Sensor()
#    print ""
#    print "Points"
#    print s.points
#    print "O ", s.origin
#    print "X ", s.x
#    print "Y ", s.y
#    print "Z ", s.z
#    print "Center ", s.parametricToPhysical(np.array([[0.5,0.5],[1,1],[0,0]]))
#    print "FY,FZ  ", s.parametricToPhysical(np.array([1,1]))
#    print "OY,OZ  ", s.parametricToPhysical(np.array([0,0]))
#
#
#    tic = Utils.Tictoc()
#    repeats     = int(2e5)
#    msze        = 15e-6
#    mengy       = 1e-14 / (15e-6 / msze)**2
#    sze         = msze  * np.abs(1 + 0.041   * np.random.randn(repeats))
#    energy      = mengy * np.abs(1 + 0.5   * np.random.randn(repeats))
#    pos         = 1.1*np.random.rand(repeats,2)-0.05
#    #pos         = np.array([[0,0],[1,0]])
#
#    tic.tic()
#    s.recordParticles(pos,energy,532e-9,sze)
#    tic.toc(repeats)
#    
#    print np.max(s.rawData)
#    s.createDeadPixels(1e-7)
#    
#    print "Size statistics"
#    print "min ", (np.min(sze)  / msze)
#    print "max ", (np.max(sze)  / msze)
#    print "avg ", (np.mean(sze) / msze)
#    print "std ", (np.std(sze)  / msze)
#    print "Energy statistics"
#    print "min ", (np.min(energy)  / mengy)
#    print "max ", (np.max(energy)  / mengy)
#    print "avg ", (np.mean(energy) / mengy)
#    print "std ", (np.std(energy)  / mengy)
#    s.displaySensor()       