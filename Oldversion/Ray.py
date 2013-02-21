from __future__ import division
import numpy as np
import copy
import vtk

class Ray():
    """ A class for doing general-purpose, surface based raytracing/casting 
    
    It should be initialized with its origin and initial vector, otherwise
    must use reset, because many parameters are redundant (for ease of 
    use from other classes) and some parameters are calculated only with
    expensive raytracing procedures
    
    The following parameters are stored:
    origin          - where ray starts
    originalVector  - initial ray direction
    currentPoint    - where ray is
    currentVector   - where it is pointing to
    terminal        - reference to object it is touching, otherwise None
    distance        - distance ran by the ray from its origin
    points          - path of the ray
    """
    
    def __init__(self,initpoint=np.array([0,0,0]),initvector=np.array([1,0,0])):
        self.reset(initpoint,initvector)
        
    def reset(self,initpoint=np.array([0,0,0]),initvector=np.array([1,0,0])):
        self.origin              = copy.deepcopy(initpoint)
        self.originalVector      = copy.deepcopy(initvector)
        self.currentPoint        = copy.deepcopy(initpoint)
        self.currentVector       = copy.deepcopy(initvector)
        self.color               = [0,255,0]
        self.terminal            = None
        self.distance            = 0
        self.points              = [copy.deepcopy(initpoint)]
        self.maximumRayTrace     = 100
        self.stepRayTrace        = 50
        self.bounceOnAllObjects  = True
        self.stopOnLightSource   = False
        
    def trace(self,objects):
        while (self.distance < self.maximumRayTrace) and \
              ((np.linalg.norm(self.currentVector) - 1)**2 < 1e-8):
            self.traceUpToNext(objects)

    def calculateNextVector(self):
        """ Calculate next direction for raytracing
        Following logic is used:
            if indexOfRefraction != 0 - calculates refraction
            else
               if indexOfReflection == 0 and isLightSource == 0 - beam dump, returns zero
               if indexOfReflection == 0 and isLightSource != 0 - light, does not change direction
               else - returns reflection
        """
        # If ray did not find anything, went to infinity, etc...
        if self.terminal is None:
            return
        
        # Case dump
        if self.terminal['r'] == 0 and \
           self.terminal['n'] == 0 and \
           self.terminal['issource'] == 0:
            self.currentVector = np.array([0,0,0])
            return
        
        # Case light sheet
        if self.terminal['n'] == 0 and \
           self.terminal['r'] == 0 and \
           self.terminal['issource'] != 0:
            if self.stopOnLightSource:
                self.currentVector = np.array([0,0,0])
                return
            else:
                return
            
        # Case is found an non-optical subject and will not bounce
        if (not self.bounceOnAllObjects) and (self.terminal['r'] == -1):
            self.currentVector = np.array([0,0,0])
            return
        
        # Case lens (or reflection, which will be calculated later
        #
        # formulation taken from: http://en.wikipedia.org/wiki/Snell's_law
        #
        Vi = self.currentVector / np.linalg.norm(self.currentVector)
        N  = self.terminal['normal']
        if self.terminal['n'] != 0:
            if (np.dot(Vi,N) < 0):
                # Case the ray is entering medium:
                n1 = self.terminal['n_ext']
                n2 = self.terminal['n']
            else:
                # Case the ray is exiting medium:
                n2 = self.terminal['n_ext']
                n1 = self.terminal['n'] 
                N  = -N # formulation needs inverted normal
                         
            cos_theta1 = np.dot(N,-Vi)
            root       = 1 - (1-cos_theta1**2)*(n1/n2)**2
            if root > 0:
                cos_theta2 = np.sqrt(root)
                self.currentVector = (n1/n2)*Vi + ((n1/n2)*cos_theta1 - cos_theta2)*N
                self.currentVector = self.currentVector / np.linalg.norm(self.currentVector)
                return

        # Case is reflection
        #if (np.dot(Vi,N) > 0):
        #    N = -N 
        self.currentVector = Vi - 2*N*np.dot(Vi,N)
        return        
        
    def traceUpToNext(self,objects):
        tmin                 = 1
        t                    = self.stepRayTrace
        if self.distance + t > self.maximumRayTrace:
            t = self.maximumRayTrace - self.distance
        vector               = t * self.currentVector/np.linalg.norm(self.currentVector)
        finalPoint           = (self.currentPoint + vector)
        nextPoint            = finalPoint
        self.terminal        = None 
        
        for o in objects:
            intersections = o.intersectLineWithPolygon(self.currentPoint,finalPoint)
            if intersections != None:
                tnext = intersections.keys()
                tnext.sort()
                if len(tnext) > 0:
                    if tnext[0] < tmin:
                        nextPoint      = copy.deepcopy(intersections[tnext[0]]['coords'])
                        tmin           = copy.deepcopy(tnext[0])
                        self.terminal  = copy.deepcopy(intersections[tnext[0]])
                #print intersections[tmin]

        self.currentPoint  = nextPoint
        self.distance      = self.distance +  tmin*np.linalg.norm(vector)
        self.points.append(copy.deepcopy(self.currentPoint))
        self.calculateNextVector()
                
    def vtkActor(self):
        """
        Returns an object of type vtkLODActor for rendering within a VTK pipeline
        """
        me  = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        cts = vtk.vtkCellArray()
            
        for n in range(len(self.points)):
            pts.InsertPoint(n,self.points[n][0],self.points[n][1],self.points[n][2])
            
        for n in range(1,len(self.points)):
            cts.InsertNextCell(2)
            cts.InsertCellPoint(n-1)
            cts.InsertCellPoint(n)
              
        me.SetPoints(pts)
        me.SetLines(cts)
                          
        dataMapper = vtk.vtkPolyDataMapper()
        dataMapper.SetInput(me)

        dataActor =vtk.vtkLODActor()
        dataActor.SetMapper(dataMapper)
        if self.color is not None:
            dataActor.GetProperty().SetColor(self.color[0],self.color[1],self.color[2])
            
        return [dataActor]
             
if __name__=="__main__":
    """
    Code for unit testing ray class
    """
    
    import vtk
    import Object
    import Utils
    from pprint import pprint
    
    tic = Utils.Tictoc()
       
    print ""       
    print "###         CHECK RAYTRACING BASIC FUNCTIONALITIES               ###"
    r = Ray(np.array([0,0,0]), np.array([1,0,0]))
    r.maximumRayTrace = 3.14159265
    r.stepRayTrace    = 5
    r.trace([])
    print "maximum length criterion"
    assert len(r.points) == 2
    assert np.linalg.norm(r.points[1] - np.array([3.14159265,0,0])) < 1e-6
    
    print "final vector"
    assert np.linalg.norm(r.currentVector - np.array([1,0,0])) < 1e-6

    print "checking reflection"
    mesh                = Object.Object()
    points              = np.array([[3, 0.5, 1], [3, 0.5,-1],
                                    [2,-0.5,-1], [2,-0.5, 1]])
    cts                 = np.array([[0,1,3],[1,2,3]])
    mesh.points         = points
    mesh.connectivity   = cts
    mesh.indexOfRefraction            = 0
    mesh.indexOfRefractionAmbient     = 1
    mesh.indexOfReflection            = 1
    mesh.isLightSource                = 0
    
    r.reset(np.array([0,0,0]), np.array([1,0,0]))
    r.maximumRayTrace = 5
    r.stepRayTrace    = 5
    r.trace([mesh])
    assert np.linalg.norm(r.points[1] - np.array([2.5,0,0])) < 1e-6
    assert np.linalg.norm(r.points[2] - np.array([2.5,2.5,0])) < 1e-6
    assert np.linalg.norm(r.currentVector - np.array([0,1,0])) < 1e-6
    mesh.rotateAroundAxis(180,mesh.z,np.array([2.5,0,0]))
    r.reset(np.array([0,0,0]), np.array([1,0,0]))
    r.maximumRayTrace = 5
    r.stepRayTrace    = 5
    r.trace([mesh])
    assert np.linalg.norm(r.points[1] - np.array([2.5,0,0])) < 1e-6
    assert np.linalg.norm(r.points[2] - np.array([2.5,2.5,0])) < 1e-6
    assert np.linalg.norm(r.currentVector - np.array([0,1,0])) < 1e-6
    mesh.rotateAroundAxis(180,mesh.z,np.array([2.5,0,0]))
    
    print "checking pass-through"
    mesh.indexOfRefraction            = 0
    mesh.indexOfRefractionAmbient     = 1
    mesh.indexOfReflection            = 0
    mesh.isLightSource                = 1
    
    r.reset(np.array([0,0,0]), np.array([1,0,0]))
    r.maximumRayTrace = 5
    r.stepRayTrace    = 5
    r.trace([mesh])
    assert np.linalg.norm(r.points[2] - np.array([5,0,0])) < 1e-6
    assert np.linalg.norm(r.currentVector - np.array([1,0,0])) < 1e-6
    
    print "checking refraction"
    mesh.indexOfRefraction            = 1.41421356237
    mesh.indexOfRefractionAmbient     = 1
    mesh.indexOfReflection            = 0
    mesh.isLightSource                = 0
    
    r.reset(np.array([0,0,0]), np.array([1,0,0]))
    r.maximumRayTrace = 5
    r.stepRayTrace    = 5
    r.trace([mesh])
    assert np.linalg.norm(r.points[2] - np.array([4.91481456572,-0.64704761275,0])) < 1e-6
    assert np.linalg.norm(r.currentVector - np.array([0.96592582628,-0.2588190451,0])) < 1e-6
    
    print "checking critical angle - 1 - ray refracts parallel to surface"
    mesh.indexOfRefraction            = 0.70710679
    mesh.indexOfRefractionAmbient     = 1
    mesh.indexOfReflection            = 0
    mesh.isLightSource                = 0
    
    r.reset(np.array([0,0,0]), np.array([1,0,0]))
    r.maximumRayTrace = 5
    r.stepRayTrace    = 5
    r.trace([mesh])
    assert np.linalg.norm(r.points[2] - np.array([4.26776695297,1.76776695297,0])) < 1e-3
    assert np.linalg.norm(r.currentVector - np.array([0.70710678118,0.70710678118,0])) < 1e-3
    
    print "checking critical angle - 2 - ray reflects entirely"
    mesh.indexOfRefraction            = 0.70710677
    mesh.indexOfRefractionAmbient     = 1
    mesh.indexOfReflection            = 0
    mesh.isLightSource                = 0
    
    r.reset(np.array([0,0,0]), np.array([1,0,0]))
    r.maximumRayTrace = 5
    r.stepRayTrace    = 5
    r.trace([mesh])
    assert np.linalg.norm(r.points[2] - np.array([2.5,2.5,0])) < 1e-6
    assert np.linalg.norm(r.currentVector - np.array([0,1,0])) < 1e-6
    
    print "done, plotting"
    Utils.displayScenario([r,mesh],True,False)
    