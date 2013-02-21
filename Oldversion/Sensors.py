from __future__ import division
import numpy as np
import Object
import Planes
import Utils
import matplotlib.pyplot as plt 
import vec

class Sensor(Planes.Plane):
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
        self.length                 = 0.0118 # by definition, the column dimension
        self.heigth                 = 0.0089  # the row dimension
        # self.length = 0.036
        # self.heigth = 0.024
        
                                               # ROW         # COLUMN
                                               #0.0089         0.0118
        self.resolution             = np.array([1200,         1600])
        self.pixelSize              = np.array([7.40,         7.40])*1e-6
        self.fillRatio              = np.array([0.75,         0.75])
        self.fullWellCapacity       = 40e3
        self.quantumEfficiency      = 0.5
        self.bitDepth               = 14
        self.backgroundMeanLevel    = 10
        self.backgroundStdNoise     = 10
        self.rawData                = None
        self.saturationData         = None
        self.deadPixels             = None
        self.hotPixels              = None
        self.virtualData            = None
        Planes.Plane.__init__(self,self.length,self.heigth)
        self.clear()
        
    def parametricToPixel(self,coordinates):
        """
        coords = [y,z] (in parametric 0..1 space)
        
        returns:
        [row column] - fractional position in sensor pixels
        
        DOES NOT CHECK IF OUTSIDE SENSOR BOUNDARIES!!!
        """
        return coordinates*self.resolution
        
    def displaySensor(self):
        imgplot = plt.imshow(self.readSensor()/(-1+2**self.bitDepth))
        imgplot.set_cmap('jet')
        imgplot.set_interpolation('none')
        plt.show()        
               
    def createDeadPixels(self,probability):
        """
        Creates a dead/hot pixel mapping for the sensor reading simulation
        """
        map = np.random.rand(self.resolution[0],self.resolution[1])
        self.deadPixels = map > probability*0.5
        self.hotPixels  = map > 1 - probability*0.5
        
    def clear(self):
        """
        Initializes CCD with gaussian noise. This logic was created
        by Lecordier and Westerweel on the SIG
        """
        self.rawData = self.backgroundMeanLevel + \
                       self.backgroundStdNoise*   \
                       np.sqrt(-2*np.log(1-np.random.rand(self.resolution[0],    \
                                                          self.resolution[1])))*  \
                       np.cos(np.pi*np.random.rand(self.resolution[0],    \
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
            
    def toSensorCoordinates(self,coords):   
        """
        coords = [x,y,z] (in real space)
        
        returns:
        [row column] - fractional position in sensor pixels
        
        DOES NOT CHECK IF OUTSIDE SENSOR BOUNDARIES!!!
        """
        return self.physicalToParametric(coords)*self.resolution
       
    def recordParticle(self,coords,energy,wavelength,diameter):
        """
        coords     - [y,z] parametric coordinates of the recorded point
        energy     - J -   energy which is captured by the lenses
        wavelength -       illumination wavelength
        diameter   - m -   particle image diameter
        
        Writes the diffraction-limited image of a particle to the
        sensor. This logic was created by Lecordier and 
        Westerweel on the SIG
        
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
           
          When particle image is partially out of the image limits, the computation
          is done over a partially useless domain, but remains correct
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
        
        The program performs an integration of a 2D gaussian distribution over the
        sensitive areas of the pixel (defined by the fill ratio). 
        
        The static method erfnorm was defined because scipy was not available
        
        This function is NOT suited for calculating more than one particle at a 
        time.
        
        """
        if sum((coords < 0) + (coords > 1)):
            return
        # Classical formula  E =    h*c        h = 6.62607e-34 J*s
        #                        ---------     c = 299 792 458 m/s
        #                          lambda
        photonEnergy = 6.62607e-34*299792458/wavelength
        totalPhotons = energy / photonEnergy
        # print "Total Photons ", totalPhotons
        if totalPhotons < 2:
            return
        #
        pixels   = self.parametricToPixel(coords) 
        # this masksize guarantees that the particle image will not be cropped, when
        # the particle size is too small
        masksize = np.round(2.25*diameter/self.pixelSize) 
        masksize = (masksize > 3)*masksize + 3*(masksize <= 3)
        # masksize = np.round(10*diameter/self.pixelSize) 
        # Defines the anchor position (cf. documentation above)
        anchor   = np.round(pixels - masksize/2) 
        # Important to verify if anchor will not force an incorrect matrix addition
        anchor   = anchor * (anchor >= 0) * (anchor + masksize <= self.resolution) + \
                        (np.array([0,0])) * (anchor < 0) + \
                        (anchor-masksize) * (anchor + masksize > self.resolution)
        [X,Y] = np.meshgrid(range(int(anchor[0]),int(masksize[0]+anchor[0])),
                            range(int(anchor[1]),int(masksize[1]+anchor[1])))
        # Magic Airy integral, in fact this is a 2D integral on the sensitive
        # area of the pixel (defined by the fill ratio)
        s = (diameter/self.pixelSize)*(0.44/1.22)
        gx0 = Sensor.erfnorm(((X-pixels[0]) - 0.5*self.fillRatio[0])*2/s[0])
        gx1 = Sensor.erfnorm(((X-pixels[0]) + 0.5*self.fillRatio[0])*2/s[0])
        gy0 = Sensor.erfnorm(((Y-pixels[1]) - 0.5*self.fillRatio[1])*2/s[1])
        gy1 = Sensor.erfnorm(((Y-pixels[1]) + 0.5*self.fillRatio[1])*2/s[1])

        level = (gx1-gx0)*(gy1-gy0)*totalPhotons*self.quantumEfficiency
        # print "Level matrix, rows: ", np.size(level,0), " columns: ", np.size(level,1)
        # print level
        self.rawData[anchor[0]:(anchor[0]+masksize[0]),anchor[1]:(anchor[1]+masksize[1])] = \
            self.rawData[anchor[0]:(anchor[0]+masksize[0]),anchor[1]:(anchor[1]+masksize[1])] + \
            level
        
    @staticmethod
    def erfnorm(input):
        """
        The algorithm of the error function comes from 
        
        Handbook of Mathematical Functions, formula 7.1.26.
        http://www.math.sfu.ca/~cbm/aands/frameindex.htm
        
        Vectorized, in order to improve performance
        
        Normalization (return line) used to calculate the integral of
        the Airy function. This logic was created by Lecordier and 
        Westerweel on the SIG.
        """
        # save the sign of x
        # sign = np.sign(input)
        x    = np.abs(input)

        # constants
        # a1 =  0.254829592
        # a2 = -0.284496736
        # a3 =  1.421413741
        # a4 = -1.453152027
        # a5 =  1.061405429
        # p  =  0.3275911

        # A&S formula 7.1.26
        # t = 1.0/(1.0 + p*x)
        t = 1.0 / (1.0 + 0.3275911*x)
        # y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
        y = 1.0 - (((((1.061405429*t -1.453152027)*t) \
                       + 1.421413741)*t -0.284496736)*t \
                       + 0.254829592)*t*np.exp(-x*x)
        # y = y*sign
        y = y*np.sign(input)
        return ((y/1.414213562373)+1)/2     
        
if __name__=='__main__':
    s = Sensor()
    s.translate(np.array([0,0.5,0.5]))
    s.rotateAroundAxis(45,s.z,s.origin)
    print ""
    print "Points"
    print s.points
    print "O ", s.origin
    print "X ", s.x
    print "Y ", s.y
    print "Z ", s.z
    print "Center ", s.parametricToPhysical(np.array([[0.5,0.5],[1,1],[0,0]]))
    print "FY,FZ  ", s.parametricToPhysical(np.array([1,1]))
    print "OY,OZ  ", s.parametricToPhysical(np.array([0,0]))
    
    coords = []
    for n in range(5):
        y = vec.linspace(0,1,2**n)
        z = y
        for yc in y:
            for zc in z:
                coords.append(s.parametricToPhysical(np.array([yc,zc])))
    coords = np.array(coords)
    fig = plt.figure()
    
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt
    ax = fig.gca(projection='3d')
    ax.scatter(coords[:,0],coords[:,1],coords[:,2])
    plt.show()
        
    # print "answer ", s.intersectLineWithPolygon(s.x,np.array([-1e-10,     0,   0]))
    # print "answer ", s.intersectLineWithPolygon(s.x,np.array([0.0001, 0,   0]))
    # print "size sensor, lines: ", np.size(s.rawData,0), " columns: ", np.size(s.rawData,1)
    
    # initpoint   = np.array([0,s.dimension[0],s.dimension[1]])*0.5*0
    # multiplier  = np.array([0,s.dimension[0],s.dimension[1]])*0.05
    
    # import Utils
    # tic = Utils.Tictoc()
    
    # tic.reset()
    # for n in range(1000):
        # pos  = np.random.rand(3)*multiplier - initpoint
        # sze = 15e-6
        # s.recordParticle(pos,5e-14,sze)
    # tic.toctask(1000)
    # print np.max(s.rawData)
    # s.createDeadPixels(3e-6)
    
    # s.displaySensor()