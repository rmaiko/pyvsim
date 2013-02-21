from __future__ import division
import numpy as np
import copy
import vtk
import Object
import pprint
import time
import threading
import vec

class PIVSimWindow(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)        
        self.footText       = "PIVSim ver. 0.0"
        self.windowTitle    = "PIVSim Visualization window - powered by VTK"
        self.windowColor    = [0,0.7,0.9]
        self.windowSize     = [800,800]
        self.actorlist      = None
        
        self.rend           = vtk.vtkRenderer()
        self.window         = vtk.vtkRenderWindow()
        self.interactor     = vtk.vtkRenderWindowInteractor()
        self.legend         = vtk.vtkTextActor()
        
        self.rend.SetBackground(self.windowColor[0],
                                self.windowColor[1],
                                self.windowColor[2])

        
        self.window.SetSize(self.windowSize[0],self.windowSize[0])
        self.window.AddRenderer(self.rend)

        self.interactor.SetRenderWindow(self.window)   

        self.legend.GetTextProperty().SetFontSize(12)
        self.legend.SetPosition2(0,0)
        self.legend.SetInput(self.footText)
        self.rend.AddActor(self.legend)  
    
    def addActors(self,actorlist):
        self.actorlist = actorlist
        for actor in actorlist:
            self.rend.AddActor(actor)
            
    def run(self):
        self.window.Render()
        self.window.SetWindowName(self.windowTitle)
        self.interactor.Start()

class Tictoc:
    def __init__(self):
        self.begin = time.clock()
        
    def reset(self):
        self.begin = time.clock()
        
    def toc(self):
        t = (time.clock()-self.begin)
        print "Elapsed time: %f seconds" % t
        return t
        
    def toctask(self,n):
        t = (time.clock()-self.begin)
        print "Can execute: %f calculations / second" % (n/t)
        return n/t

def readSTL(filename):
    STLReader   = vtk.vtkSTLReader()

    STLReader.SetFileName(filename)
    STLReader.Update()
    polydata = STLReader.GetOutput()    
    points      = polydata.GetPoints()
    
    pts = []
    cts = []
    for n in range(points.GetNumberOfPoints()):
        pts.append(points.GetPoint(n))
    
    for n in range(polydata.GetNumberOfCells()):
        temp = vtk.vtkIdList()
        polydata.GetCellPoints(n,temp)
        cts.append([temp.GetId(0),temp.GetId(1),temp.GetId(2)])
        
    obj         = Object.Object()
    obj.points          = np.array(pts)
    obj.connectivity    = np.array(cts)
    
    return obj
    
def pointInHexa(p,hexapoints):
    """
    Taking a set of points defining a hexahedron in the conventional order,
    this function tests of the point is inside this hexahedron by:
    
    For each face:
        - calculate normal pointing outwards
        - verify if point is "behind" the plane defined by the face
        
    Returns 1 if inside or on the face, 0 otherwise
    """
    # Definition of triangles using the hexa points
    l = np.array([[1,4,0],
                  [3,1,0],
                  [4,3,0],
                  [5,2,6],
                  [7,5,6],
                  [2,7,6]])
    
    if p.ndim > 1:
        truthtable = np.ones(len(p))
    else:
        truthtable = 1
               
    for n in range(6):
        vout = np.cross(hexapoints[l[n,0]]-hexapoints[l[n,2]],
                        hexapoints[l[n,1]]-hexapoints[l[n,2]])
        truthtable = truthtable * (vec.dot(p - hexapoints[l[n,2]],vout) < 0)

    return truthtable        
        
def quadArea(p1,p2,p3,p4):
    return triangleArea(p1,p2,p3) + triangleArea(p1,p3,p4)
   
def rotateVector(x,angle,axis,deg=True):
    """
    This implementation uses angles in degrees. The algorithm is the vectorial
    formulation of the Euler-Rodrigues formula as found in:
    http://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_parameters
    """
    if deg:
        angle = np.deg2rad(angle)
    a = np.cos(angle/2)
    w = axis*np.sin(angle/2)
    return x + 2*a*np.cross(w,x) + 2*np.cross(w,np.cross(w,x))
    
def rotatePoints(points,angle,axis,origin):
    """
    Wrap-around Euler-Rodrigues formula for rotating a point cloud
    """
    if len(points) > 0:
        return rotateVector(points-origin,angle,axis) + origin
    else:
        return points
   
def triangleArea(p1,p2,p3):
    """
    Given three points in space, returns the triangle area. Assumes:
    1 - points are given as numpy arrays (crashes if not met)
    
    Algorithm:
        v1 = p2 - p1
        v2 = p3 - p1
        return 0.5*np.linalg.norm(np.cross(v1,v2))
    """ 
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5*vec.norm(np.cross(v1,v2))
       
def triangleNormal(p1,p2,p3):
    """
    Given three points in space, returns the triangle normal. Assumes:
    1 - points are given as numpy arrays (crashes if not met)
    2 - points are given counterclockwise (negative result if not met)
    
    Algorithm:
        v1 = p2 - p1
        v2 = p3 - p1
        N = np.cross(v1,v2)
        return N/np.linalg.norm(N)
    """
    v1 = p2 - p1
    v2 = p3 - p1
    N = np.cross(v1,v2)
    return vec.normalize(N)

def barycentricCoordinates(p,p1,p2,p3):
    """
    Given:
    1 - p : point in space
    2 - p1,p2,p3 : points space representing triangle
    Returns:
    [lambda1,lambda2,lambda3] the barycentric coordinates of point p with 
    respect to the defined triangle
    Assumes:
    1 - points are given as numpy arrays (crashes if not met)
    
    Algorithm:
        area        = Object.triangleArea(p1,p2,p3)
        lambda1     = Object.triangleArea(p,p2,p3)
        lambda2     = Object.triangleArea(p,p1,p3)
        lambda3     = Object.triangleArea(p,p1,p2)
        return np.array([lambda1,lambda2,lambda3])/area
    """
    area        = triangleArea(p1,p2,p3)
    lambda1     = triangleArea(p,p2,p3)
    lambda2     = triangleArea(p,p1,p3)
    lambda3     = triangleArea(p,p1,p2)
    return np.array([lambda1,lambda2,lambda3])/area
       
def EQBInterpolation(p,p1,p2,p3,p4,v1,v2,v3,v4): 
    """
    Performs barycentric interpolation extended to the case of a planar quad
    """
    Su = triangleArea(p,p4,p3)
    Sr = triangleArea(p,p3,p2)
    Sd = triangleArea(p,p1,p2)
    Sl = triangleArea(p,p1,p4)
    den = (Su + Sd)*(Sr + Sl)
    c1 = Sr*Su/den
    c2 = Sl*Su/den
    c3 = Sl*Sd/den
    c4 = Sr*Sd/den
    return vec.listTimesVec(c1,v1) + vec.listTimesVec(c2,v2) + \
           vec.listTimesVec(c3,v3) + vec.listTimesVec(c4,v4)
    
def displayScenario(objectsList,displayAxes=True,adaptAxes=True,window=None):
    # tic = Tictoc()
    # tic.reset()
    actorlist = []
    if displayAxes:
        axesActor = vtk.vtkAxesActor()
        axesActor.SetShaftTypeToLine()
        
        actorlist.append(axesActor)
    
    for o in objectsList:
        actorlist.extend(o.vtkActor())

    if displayAxes and adaptAxes:      
        maxdimensions = [0,0,0]
        for o in objectsList:
            try:
                bounds = o.getBounds()
                for n in range(3):
                    if maxdimensions[n] < bounds[2*n+1]-bounds[2*n]:
                        maxdimensions[n] = bounds[2*n+1]-bounds[2*n]
            except:
                pass
        axesActor.SetTotalLength(maxdimensions)
    if window == None:
        window = PIVSimWindow()
        window.addActors(actorlist)
        window.start()
        return window
    else:
        window.removeAllActors()
        window.addActors(actorlist)
    # print "Time to render"
    # tic.toc()
    

    

 
if __name__=="__main__":
    """
    Code for unit testing class
    """
    
    import vtk
    import Object
    import Utils
    import pprint
    import Ray
    
    print "### File Util has no Unit testing ###"
    
    
    
    # tic = Util.Tictoc.Tictoc()
    
    # tic.reset()
    # lasersheet = Utils.readSTL('test.stl')
    # print "Time to read %i polys" % len(lasersheet.connectivity)
    # tic.toc()
    
    # lasersheet.indexOfRefraction            = 2
    # lasersheet.indexOfRefractionAmbient     = 1
    # lasersheet.indexOfReflection            = 0
    # lasersheet.isLightSource                = 0

    # #
    # # Definition of the camera
    # #

    # # pinhole location
    # p = [0,+50,0] 
    # # pixel location (origin for raytracing)
    # s = []
    # xrange = 3
    # yrange = 3
    # for x in range(-xrange,xrange):
        # for y in range(-yrange,yrange):
            # s.append([x/xrange, 55, y/yrange])
            
    # actorlist  = []
    # mapperlist = []
    # rlist      = []
    
    # axesActor = vtk.vtkAxesActor()
    # actorlist.append(axesActor)
       
    # actorlist.insert(0,lasersheet.vtkActor())   
    
    # tic.reset()
    # for px in s:  
        # v = np.array(p)-np.array(px)
        # v = v / np.sqrt(np.dot(v,v))
        
        # rlist.insert(0,Ray.Ray(p,v))
        # rlist[0].bounceOnAllObjects = True
        # rlist[0].trace([lasersheet])

        # actorlist.insert(0,rlist[0].vtkActor())      
    
    # tic.toc()
    
    # rend = vtk.vtkRenderer()
    # rend.SetBackground(0.0,0.0,0.6)
    # for actor in actorlist:
        # rend.AddActor(actor)

    # window = vtk.vtkRenderWindow()
    # window.AddRenderer(rend)

    # interactor = vtk.vtkRenderWindowInteractor()
    # interactor.SetRenderWindow(window)

    # window.Render()
    # interactor.Start()