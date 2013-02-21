from __future__ import division
import vtk
import numpy as np
import copy
import Object
import Ray
import math
import Utils
import vec
import Sensors
import Objectives
import Ray

class Camera(Object.Assembly):
    CIRCLE_OF_CONFUSION_DIAMETER = 29e-6 # 29 microns is used in photographic literature
    MAPPING_SAMPLING_POINTS      = 200
    def __init__(self):
        # Properties inherited from Object
        Object.Assembly.__init__(self)
        self.indexOfRefraction          = 0
        self.indexOfRefractionAmbient   = 0
        self.indexOfReflection          = -1
        self.isLightSource              = 0
        self.points                     = np.array([])
        self.connectivity               = np.array([[4,7,5],[7,6,5], # normal -z
                                                    [1,2,3],[1,3,0], # normal +z
                                                    [2,6,3],[7,3,6], # normal -y
                                                    [4,5,1],[1,0,4], # normal +y
                                                    [6,1,5],[6,2,1], # normal -x
                                                    [4,0,7],[0,3,7]]) # normal +x
        self.normals                    = None
        self.sensor                     = Sensors.Sensor()
        self.sensorPosition             = -0.017526
        #self.sensorPosition             = -0.044
        self.objective                  = Objectives.Objective()
        self.maximumImagingDistance     = 10
        # Plotting properties
        self.color                      = [0,0.2,1]
        self.opacity                    = 1.0
        self.width                      = 0.084
        self.heigth                     = 0.066
        self.length                     = 0.175
        self.raylist                    = None
        # Mapping properties
        self.mapping                    = None
        self.error                      = None
        self.sensorSamplingCenters      = None
        self.sensorSamplingPositions    = None
        self.physicalSamplingCenters    = None
        self.physicalSamplingPositions  = None
        self.objectList                 = [self.sensor,self.objective]
        self.positionSubcomponents()
     
    def positionSubcomponents(self):
        """
        This method is a shortcut to define the initial position of the camera,
        there is a definition of the self.points property, which represents the
        camera body, and the initial positioning of the sensor and the objective.
        """
        # Really manual definition of the camera points, please note that its 
        # side is pointing to the y axis, to comply with the sensor definition
        self.points = np.array( \
        [-1*self.x*self.length-1*self.y*0.5*self.heigth-1*self.z*0.5*self.width,
         -1*self.x*self.length+1*self.y*0.5*self.heigth-1*self.z*0.5*self.width,
         -1*self.x*self.length+1*self.y*0.5*self.heigth+1*self.z*0.5*self.width,
         -1*self.x*self.length-1*self.y*0.5*self.heigth+1*self.z*0.5*self.width,
         +0*self.x*self.length-1*self.y*0.5*self.heigth-1*self.z*0.5*self.width,
         +0*self.x*self.length+1*self.y*0.5*self.heigth-1*self.z*0.5*self.width,
         +0*self.x*self.length+1*self.y*0.5*self.heigth+1*self.z*0.5*self.width,
         +0*self.x*self.length-1*self.y*0.5*self.heigth+1*self.z*0.5*self.width])
        self.points = self.points + self.origin
        
        # Sensor positioning
        self.sensor.translate(self.origin - self.sensor.origin)
        self.sensor.alignTo(self.x,self.y)
        # Flange focal distance adjustment
        self.sensor.translate(self.x*self.sensorPosition)
        
        # Objective positioning
        self.objective.alignTo(self.x,self.y)
        self.objective.translate(self.origin - self.objective.origin)
        
    def destroyData(self):
        self.mapping                    = None
        self.error                      = None
        self.physicalSamplingCenters    = None
        self.sensorSamplingCenters      = None
        self.raylist                    = None
        #self.positionSubcomponents()
        Object.Object.destroyData(self)
    
    def calculateMapping(self,environment):
        """
        This method populates the self.mapping tensor. This is done on the following
        way:
        0 - the sensor is meshed in a MAPPING_SAMPLING_POINTS**2 grid
        
        1 - for each mesh point, rays are casted through the environment, 
            and the final positions are stored, as well as the distance ran 
            by the ray
            
        2 - the submatrix M is defined as:
            [x y z 1] * M = [s1 s2]
            where:
            [x y z] = physical coordinates of the position the ray ends
            [s1 s2] = sensor parametric coordinates (where the ray starts)
            The matrix is calculated using the 4 points on the extremity of the cell
            
        3 - the matrix M is stored at the self.mapping property
        
        The property self.error is calculated with the
        """
        #
        # Ad hoc stuff - must define a better way to determine the number of
        # casted rays (most of times 4 is enough)
        steps = self.MAPPING_SAMPLING_POINTS
        c = vec.linspace(0,1,steps)     
        [Z,Y] = np.meshgrid(c,c)
        # Calculate centerpoints
        Yc    = (Y[1:,1:] + Y[:-1,:-1])/2
        Zc    = (Z[1:,1:] + Z[:-1,:-1])/2
        
        M     = np.zeros((np.size(Yc,0),np.size(Yc,1),4,2))
        error = np.zeros((np.size(Yc,0),np.size(Yc,1)))
        
        
        Xr = np.zeros((np.size(Y,0),np.size(Y,1)))
        Yr = np.zeros((np.size(Y,0),np.size(Y,1)))
        Zr = np.zeros((np.size(Y,0),np.size(Y,1)))
        
        D  = np.zeros((np.size(Y,0),np.size(Y,1)))

        #
        # Cast rays and store endpoints
        #
        for i in range(np.size(Y,0)):
            for j in range(np.size(Y,1)):
                Xsensor = self.sensor.parametricToPhysical(np.array([Y[i,j],Z[i,j]]))
                r = self.objective.ray(Xsensor)
                r.stopOnLightSource = True
                r.maximumRayTrace = self.maximumImagingDistance
                r.stepRayTrace    = self.maximumImagingDistance
                
                r.trace(environment)

                [Xr[i,j],Yr[i,j],Zr[i,j]] = r.currentPoint
                D[i,j] = r.distance
                
        Xrc = (Xr[1:,1:] + Xr[:-1,:-1])/2
        Yrc = (Yr[1:,1:] + Yr[:-1,:-1])/2
        Zrc = (Zr[1:,1:] + Zr[:-1,:-1])/2
        #
        # Calculates the transformation matrixes
        #
        for i in range(np.size(Yc,0)):
            for j in range(np.size(Yc,1)):
                # The vectors are extended in order to account for a rotation +
                # translation matrix
                realpoints = np.array([[Xr[i  ,j  ],Yr[i  ,j  ],Zr[i  ,j  ],1],
                                       [Xr[i+1,j  ],Yr[i+1,j  ],Zr[i+1,j  ],1],
                                       [Xr[i+1,j+1],Yr[i+1,j+1],Zr[i+1,j+1],1],
                                       [Xr[i  ,j+1],Yr[i  ,j+1],Zr[i  ,j+1],1]])

                sensorpoints = np.array([[Y[i  ,j  ],Z[i  ,j  ]],
                                         [Y[i+1,j  ],Z[i+1,j  ]],
                                         [Y[i+1,j+1],Z[i+1,j+1]],
                                         [Y[i  ,j+1],Z[i  ,j+1]]]) 
                regression = np.linalg.lstsq(realpoints,sensorpoints)
                M[i,j,:,:] = regression[0]
                error[i,j] = (np.sum(
                             (np.dot(np.array([Xrc[i,j],Yrc[i,j],Zrc[i,j],1]),M[i,j,:,:]) - 
                             np.array([Yc[i,j],Zc[i,j]]))**2))**0.5
                # error[i,j] = np.max(np.abs(np.dot(realpoints,M[i,j])-sensorpoints))

        self.mapping                    = M
        self.distanceToImagingPlane     = D
        self.sensorSamplingCenters      = np.array([Yc,Zc])
        self.sensorSamplingPositions    = np.array([Y,Z])
        self.physicalSamplingCenters    = np.array([Xrc,Yrc,Zrc])
        self.physicalSamplingPositions  = np.array([Xr,Yr,Zr])
        self.error                      = error
       
    def mappingMatrix(self,p):
        """ 
        Returns the mapping matrix, and the i,j position in the mapping tensor
        for a given position p in space
        
        The algorithm uses nearest-neighbor interpolation when the mapping is
        defined as a piecewise linear function
        """
        distances = (p[0] - self.physicalSamplingCenters[0,:,:])**2 + \
                    (p[1] - self.physicalSamplingCenters[1,:,:])**2 + \
                    (p[2] - self.physicalSamplingCenters[2,:,:])**2
        (i,j) = np.nonzero(distances == distances.min())            
        i = i[0]
        j = j[0]
        return [self.mapping[i,j,:,:],i,j]
       
    def physicalToSensorParametric(self,p):
        """
        When the mapping is available, this function translated a coordinate in
        the physical world (preferrably in the laser sheet) to a parametric
        coordinate on the sensor surface
        """
        [M,i,j] = self.mappingMatrix(p)
        return [np.dot(np.array([p[0],p[1],p[2],1]),M), \
                self.distanceToImagingPlane[i,j]]
    
    def calculateDoF(self,environment):
        """
        This routine is used for calculating the depth of field of the camera.
        
        The formulation used here is one of similar triangles, the idea is that
        from the exit pupil to the projection point in the sensor a cone is formed,
        then a maximum sensor displacement is calculated based on the circle of
        confusion diameter and finally this is projected into real life:  
        """
        
        self.raylist = []
        
        # This calculation is needed to determine the effective focal distance
        # of the lens assembly (to be used in the thin lens equation)
        #
        # Note that there is some error (found to be small) as we use the
        # average between near and far distance to calculate the fEffective
        Dnear = (self.objective.nominalFocusDistance * self.objective.focalLength**2) / \
                (self.objective.focalLength**2 + self.objective.aperture*self.CIRCLE_OF_CONFUSION_DIAMETER*\
                (self.objective.nominalFocusDistance - self.objective.focalLength))
        Dfar  = (self.objective.nominalFocusDistance * self.objective.focalLength**2) / \
                (self.objective.focalLength**2 - self.objective.aperture*self.CIRCLE_OF_CONFUSION_DIAMETER*\
                (self.objective.nominalFocusDistance - self.objective.focalLength))
        #print "Calculated using thin lens", Dnear, Dfar
        fEffective = 1 / (1/self.objective.focalLength + 2/(Dnear+Dfar))
        points = np.array([[0,0],[1,0],[1,1],[0,1]])
        #print "Effective focal distance ", fEffective
        
        opticalAxis = self.objective.x
                                 
        for n in range(4):
            sensorPosition          = self.sensor.parametricToPhysical(points[n])
            #print "Parameter       ", points[n]
            #print "Sensor position ", sensorPosition
            vectorToExitPupil       = self.objective.exitPlaneCenter - sensorPosition
                                                  
            # This is calculated by triangle similarity
            #-------         ----------  Exit pupil
            #       \       / 
            #        \     /
            #         \---/    -> forward position (circle of confusion diameter)
            #          \ /
            #           V      -> perfect focus point
            #          / \
            #         /---\    -> backward position (circle of confusion diameter)
            
            allowableDisplacement = vec.norm(vectorToExitPupil) * \
                                    self.CIRCLE_OF_CONFUSION_DIAMETER / \
                                    (self.objective.exitPupilDiameter / self.objective.aperture)
            vectorToExitPupil     = vec.normalize(vectorToExitPupil)
            pointNear   = sensorPosition - vectorToExitPupil * allowableDisplacement
            pointFar    = sensorPosition + vectorToExitPupil * allowableDisplacement
            vNear       = self.objective.exitPlaneCenter - pointNear
            vFar        = self.objective.exitPlaneCenter - pointFar
            dNear       = vec.dot(vNear,opticalAxis)
            dFar        = vec.dot(vFar ,opticalAxis)
            dNearReal = (fEffective * dNear) / (dNear - fEffective)
            dFarReal  = (fEffective * dFar ) / (dFar  - fEffective)
            #print "Calculated using theory    ", dNearReal, dFarReal
            # When imaging at infinity, the result for dFarReal yields a negative
            # number, must correct it
            if dFarReal < 0:
                dFarReal = self.maximumImagingDistance
            nearRayLength  = dNearReal * vec.norm(vNear) / dNear
            farRayLength   = dFarReal  * vec.norm(vFar ) / dFar
            fieldRayLength = farRayLength - nearRayLength
            #
            # Generate ray in the unsharp region
            #
            self.raylist.insert(0,self.objective.ray(self.sensor.parametricToPhysical(points[n])))
            self.raylist[0].maximumRayTrace = nearRayLength
            self.raylist[0].color = [1,0,0]
            self.raylist[0].trace(environment)
            #
            # Generate ray in the sharp region
            #
            self.raylist.insert(0,Ray.Ray(self.raylist[0].currentPoint,self.raylist[0].currentVector))
            self.raylist[0].maximumRayTrace = fieldRayLength
            self.raylist[0].color = [0,1,0]
            self.raylist[0].trace(environment)
            #
            # Generate ray after sharp region (useful to check field of view
            # regardless of sharpness)
            #
            if (self.maximumImagingDistance - fieldRayLength) > 0:
                self.raylist.insert(0,Ray.Ray(self.raylist[0].currentPoint,self.raylist[0].currentVector))
                self.raylist[0].maximumRayTrace = self.maximumImagingDistance - fieldRayLength
                self.raylist[0].color = [1,0,0]
                self.raylist[0].trace(environment)
        
    def recordData(self,dataPlane,binning=1):
        ndim  = dataPlane.ndim
        print ""
        print "Msize", np.size(self.mapping,0), np.size(self.mapping,1), np.size(self.mapping,2), np.size(self.mapping,3)
        print "Esize", np.size(self.error,0), np.size(self.error,1)
        stepy = 1 / (self.MAPPING_SAMPLING_POINTS - 2)
        stepz = stepy
               
        ysteps = self.sensor.resolution[0]
        zsteps = self.sensor.resolution[1]
        if binning != 1:
            ysteps = int(np.round(ysteps / binning))
            zsteps = int(np.round(zsteps / binning))
        
        Yparam = np.linspace(0,1,ysteps)
        Zparam = np.linspace(0,1,zsteps)
         
        self.sensor.virtualData = np.zeros((ndim,ysteps,zsteps))
        # print "self.sensor.virtualData"
        # print self.sensor.virtualData.ndim, np.size(self.sensor.virtualData,0)
        
        for i in range(ysteps):
            print (i)/(ysteps)
            for j in range(zsteps):
                p       = np.array([Yparam[i],Zparam[j]])
                pi      = np.floor(Yparam[i] / stepy)
                pj      = np.floor(Zparam[j] / stepz)
                # print pi, pj, stepy, stepz
                p1      = self.sensorSamplingPositions[:,pi+0,pj+0]
                p2      = self.sensorSamplingPositions[:,pi+1,pj+0]
                p3      = self.sensorSamplingPositions[:,pi+1,pj+1]
                p4      = self.sensorSamplingPositions[:,pi+0,pj+1]
                v1      = self.physicalSamplingPositions[:,pi+0,pj+0]
                v2      = self.physicalSamplingPositions[:,pi+1,pj+0]
                v3      = self.physicalSamplingPositions[:,pi+1,pj+1]
                v4      = self.physicalSamplingPositions[:,pi+0,pj+1]
                # print "Point ", p
                # print p1, p2, p3, p4
                location = Utils.EQBInterpolation(p,p1,p2,p3,p4,v1,v2,v3,v4)
                data = dataPlane.getData(location)
                # print v1, v2
                # print v3, v4
                # print location
                # print ""
                # print i, j, data
                if data is not None:
                    self.sensor.virtualData[:,i,j] = data
        print self.sensorSamplingPositions
        print self.physicalSamplingPositions
        
    def recordParticleCloud(self,particleCloud):
        """
        Given a particle cloud, this method prints on the sensor an Airy disk for
        each particle.
        
        This method requires the mapping from calculateMapping to be performed
        """
        # recordParticle(self,coords,energy,diameter):
        
        # Project points on the sensor plane
        # V = self.objective.entrancePlaneCenter - particleCloud.particles
        # num = vec.dot(self.sensor.x, self.sensor.origin - self.objective.exitPlaneCenter)
        # den = vec.dot(self.sensor.x, V)
        # t   = num / den
        # P   = self.objective.exitPlaneCenter + vec.listTimesVec(t,V)
        
        # Calculate particle image size, WARNING, uses the following hypothesis:
        # 1 - Diffraction-limited imaging
        #
        # If this is not met, it is not possible to calculate the image using the
        # self.sensor.recordParticle method, which assumes this
        particleImageDiameter = 2.44 * particleCloud.laserWavelength * \
                                self.objective.aperture
        
        # Calculate energy, WARNING - approximation:
        # 1 - camera relatively far from the particles
        # 2 - f number low
        # If conditions are not respected must:
        # 1 - calculate scs for each particle (as angle between source and observer
        #     varies
        # 2 - instead of taking only the centerpoint to calculate the SCS, must
        #     calculate SCS for several points on the aperture and integrate
        scs = particleCloud.getDifferentialScatteringCrossSection(
                                    self.objective.entrancePupilCenter)
        E   = scs*3.141592*((self.objective.entrancePupilDiameter)**2) * \
              vec.norm(particleCloud.intensities)
             
              
        for n in range(len(particleCloud.particles)):
            location = self.physicalToSensorParametric(particleCloud.particles[n])
            energy   = E[n] / location[1]**2
            self.sensor.recordParticle(location[0],energy,532e-9,particleImageDiameter)
        
    def vtkActor(self):
        actorlist = []
        actorlist.extend(Object.Assembly.vtkActor(self))
        try:
            for r in self.raylist:
                actorlist.extend(r.vtkActor())
        except:
            pass
        return actorlist   


if __name__ == '__main__':
    import Planes
    import Utils
    import copy
    tic = Utils.Tictoc()
    c = Camera()
    c.objective.translate(np.array([0.0265,0,0]))
    c.objective.aperture                = 22
    c.objective.nominalFocusDistance    = 2.3
    c.rotateAroundAxis(90,c.x)
    #c.rotateAroundAxis(90,c.y)
    
    
    m = Planes.Mirror()
    d = Planes.Dump()
    
    m.translate(np.array([1,0,0]))
    m.rotateAroundAxis(60,m.y)
    
    d.translate(np.array([1.5,0,1]))
    d.rotateAroundAxis(90,m.y)
    
    
    d2 = copy.deepcopy(d)
    d.translate( np.array([0,0.5,0.2]))
    d2.translate(np.array([0,-0.4,0]))
    d2.rotateAroundAxis(10,m.x)
    env = [m,d,d2]
    
    print ""
    
    #c.objective.rotateAroundAxis(10,np.array([0,1,0]))
    
    c.calculateDoF(env)
    
    tic.reset()
    c.calculateMapping(env)
    tic.toc()
    
    for n in range(c.MAPPING_SAMPLING_POINTS-1):
        ans = c.physicalToSensorParametric(c.physicalSamplingCenters[:,n,0])
        print ans - c.sensorSamplingCenters[:,n,0], \
              c.physicalSamplingCenters[:,n,0], \
              c.physicalToSensorParametric(c.physicalSamplingCenters[:,n,0])
        
    Utils.displayScenario([m,d,d2,c])
    
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(c.physicalSamplingCenters[0],c.physicalSamplingCenters[1],c.physicalSamplingCenters[2])
    ax.plot_wireframe(c.sensorSamplingCenters[0],c.sensorSamplingCenters[1],c.sensorSamplingCenters[0]*0)
    
    fig = plt.figure()
    imgplot = plt.imshow(c.error)
    imgplot.set_cmap('jet')
    imgplot.set_interpolation('none')
    plt.colorbar()
    
    plt.show()  
    
    