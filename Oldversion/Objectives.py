from __future__ import division
import vtk
import Object
import math
import numpy as np
import vec
import Ray
import Utils

class Objective(Object.Object):
    def __init__(self):
        # Properties inherited from Object
        Object.Object.__init__(self)
        self.indexOfRefraction          = 0
        self.indexOfRefractionAmbient   = 0
        self.indexOfReflection          = -1
        self.isLightSource              = 0
        self.points                     = np.array([])
        self.connectivity               = np.array([])
        self.normals                    = None
        #
        # Specific properties
        #
        # Main planes model
        self.focalLength                = 0.100
        self.flangeFocalDistance        = 0.044
        self.entranceMainPlanePosition  = 0
        self.nominalFocusDistance       = 10
        # Light capture model
        self.aperture                   = 2
        self.entrancePupilDiameter      = 0.050
        self.entrancePupilLocation      = 0.0654
        # ad hoc = made 0.1, as it would be when f-number is 1
        self.exitPupilDiameter          = 0.1
        self.exitPupilLocation          = 0.0654
        self.cosMaximumImagingAngle     = math.cos(math.atan2(self.focalLength,0.0215))
        # Radial distortion model
        self.distortionParameters       = np.array([0,0,0,1])
        # Plotting parameters
        self.color                      = [0.2,0.2,0.2]
        self.opacity                    = 0.8
        self.diameter                   = 0.076
        self.length                     = 0.091
        #Calculated parameters
        self.entrancePlaneCenter              = None
        self.exitPlaneCenter                  = None
        self.mainPlanesCenter                 = None
        self.entrancePupilCenter              = None
        self.calculatePositions()
        
    def destroyData(self):
        """
        Calls calculate positions whenever the objective moves
        """
        self.calculatePositions()
        Object.Object.destroyData(self)
        
    def calculatePositions(self):
        """
        Calculate some important points, must be called whenever the lens is moved
        """
        self.exitPlaneCenter        = self.origin + \
                               self.x * (self.focalLength - self.flangeFocalDistance)
        self.exitPupilCenter        = self.origin + self.x * self.exitPupilLocation
        self.entrancePlaneCenter    = self.origin + \
                               self.x * (self.entranceMainPlanePosition - self.flangeFocalDistance)
        self.mainPlanesCenter       = (self.exitPlaneCenter + self.entrancePlaneCenter) / 2
        self.entrancePupilCenter    = self.origin + \
                                        self.x * (self.entrancePupilLocation - self.flangeFocalDistance)
    
    def ray(self,p):
        """
        Given a point (preferrably in the sensor), will return a Ray initialized
        for raytracing, if the maximum imaging angle is not respected, will return
        None
        """
        v = vec.normalize(self.exitPlaneCenter - p)
        if np.dot(self.x,v) < self.cosMaximumImagingAngle:
            return None
        return Ray.Ray(self.entrancePlaneCenter,self.lensDistortion(v))
        
    def lensDistortion(self,v):
        """
        Implementation of radial distortion model
        """ 
        Ti = math.acos(np.dot(self.x,v))
        To = np.dot(self.distortionParameters,np.array([Ti**4,Ti**3,Ti**2,Ti]))
        # print Ti, To, To-Ti
        axis = np.cross(self.x,v)
        return Utils.rotateVector(v,(To-Ti),axis,False)
        
    def vtkActor(self):
        source = vtk.vtkLineSource()
        source.SetPoint1(self.origin)
        source.SetPoint2(self.origin + self.x*self.length)
        
        tube = vtk.vtkTubeFilter()
        tube.SetInput(source.GetOutput())
        tube.SetRadius(self.diameter/2)
        tube.SetNumberOfSides(25)
        tube.CappingOn()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(tube.GetOutput())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        if self.color is not None:
            actor.GetProperty().SetColor(self.color[0],self.color[1],self.color[2]) 
        
        return [actor]
      
if __name__ == '__main__':
    import Sensors
    import Utils
    
    s = Sensors.Sensor()
    s.translate(np.array([-0.044,0,0]))
    o = Objective()
    c = np.linspace(0,1,7)
    rlist = []
    for y in c:
        for z in c:
            rlist.insert(0,o.ray(s.parametricToPhysical(np.array([y,z]))))
            rlist[0].trace([])
            # print ""
            # print o, o.entrancePlaneCenter, o.exitPlaneCenter
            # print r, r.origin, r.currentVector
    rlist.append(o)
    Utils.displayScenario(rlist)