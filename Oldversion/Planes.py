from __future__ import division
import numpy as np
import vec
import Object
import Utils

class Plane(Object.Object):
    def __init__(self,length=1,heigth=1):
        Object.Object.__init__(self)
        # Properties inherited from Object
        self.length                     = length
        self.heigth                     = heigth
        self.connectivity               = np.array([[0,1,2],
                                                    [0,2,3]])
        self.normals                    = None
        self.dimension                  = None
        self.resize(self.length,self.heigth)
        
    def resize(self,length,heigth):
        self.points     = np.array([-self.y*heigth-self.z*length,
                                    +self.y*heigth-self.z*length,
                                    +self.y*heigth+self.z*length,
                                    -self.y*heigth+self.z*length])*0.5
        self.length                 = length
        self.heigth                 = heigth
        self.dimension              = np.array([self.heigth,  self.length])
        
    def parametricToPhysical(self,coordinates):
        """
        Transforms a 2-component vector in the range 0..1 in sensor coordinates
        Normalized [y,z] -> [x,y,z] (global reference frame)
        
        Vectorization is relatively ad-hoc implemented, don't know how to improve
        that much
        """
        # Compensate for the fact that the sensor origin is at its center
        coordinates = self.dimension*(coordinates - 0.5)
        if coordinates.ndim == 1:
            return self.origin + coordinates[0] * self.y + \
                                 coordinates[1] * self.z
        else:
            return self.origin + \
                   vec.listTimesVec(coordinates[:,0],self.y) + \
                   vec.listTimesVec(coordinates[:,1],self.z)
                                 
    def physicalToParametric(self,coords):
        """
        Transforms a 3-component coordinates vector to a 2-component vector
        which value fallsin the range 0..1 in sensor coordinates
        Normalized [y,z] -> [x,y,z] (global reference frame)
        
        Vectorization is relatively ad-hoc implemented, don't know how to improve
        that much
        """
        v = coords - self.origin
        py = (vec.dot(self.y,v) / self.dimension[0]) + 0.5
        pz = (vec.dot(self.z,v) / self.dimension[1]) + 0.5
        if coords.ndim == 1:
            return np.array([py,pz])        
        else:
            return np.array([py,pz]).T

class DataPlane(Plane):
    def __init__(self,length=1,heigth=1):
        Plane.__init__(self,length,heigth)
        self._data   = None
        self.ndim    = None
        self.step0   = None
        self.step1   = None
        
    @property
    def data(self):
        return self._data
        
    @data.setter
    def data(self,data):
        if data.ndim != 3:
            raise ValueError("Data must have ndim == 3")
        self._data  = data
        self.ndim   = np.size(data,0)
        self.step0  = 1 / (np.size(data,1) - 1)
        self.step1  = 1 / (np.size(data,2) - 1)
            
    @data.deleter
    def data(self):
        del self._data
        
    def getData(self,coords):
        """
        NOT VECTORIZED!
        """
        [p0,p1] = self.physicalToParametric(coords)
        if (p0 < 0) or (p0 >= 1) or (p1 < 0) or (p1 >= 1):
            return None
        i   = np.floor(p0 / self.step0)
        j   = np.floor(p1 / self.step1)
        p   = np.array([p0,p1])
        p1  = np.array([self.step0*(i+0),self.step1*(j+0)])
        p2  = np.array([self.step0*(i+1),self.step1*(j+0)])
        p3  = np.array([self.step0*(i+1),self.step1*(j+1)])
        p4  = np.array([self.step0*(i+0),self.step1*(j+1)])
        v1  = self._data[:,i+0,j+0]
        v2  = self._data[:,i+1,j+0]
        v3  = self._data[:,i+1,j+1]
        v4  = self._data[:,i+0,j+1]
        return Utils.EQBInterpolation(p,p1,p2,p3,p4,v1,v2,v3,v4)
        
class Mirror(Plane):
    def __init__(self,length=1,heigth=1):
        Plane.__init__(self,length,heigth)
        # Properties inherited from Object
        self.indexOfRefraction          = 0
        self.indexOfRefractionAmbient   = 0
        self.indexOfReflection          = 1
        self.isLightSource              = 0
        self.color                      = [0,255,255]
        self.opacity                    = 0.8
        
class Dump(Plane):
    def __init__(self,length=1,heigth=1):
        Plane.__init__(self,length,heigth)
        # Properties inherited from Object
        self.indexOfRefraction          = 0
        self.indexOfRefractionAmbient   = 0
        self.indexOfReflection          = 0
        self.isLightSource              = 0
        self.color                      = [80,80,80]
        self.opacity                    = 0.8       
       

class Lena(DataPlane):
    def __init__(self,length=1,heigth=1):
        DataPlane.__init__(self,length,heigth)
        # Properties inherited from Object
        self.indexOfRefraction          = 0
        self.indexOfRefractionAmbient   = 0
        self.indexOfReflection          = 0
        self.isLightSource              = 0
        self.color                      = None
        self.opacity                    = None
        self.texture                    = 'lena.JPG'
        from lena import lena
        self._data                      = np.array([lena])
        self.ndim   = 1
        self.step0  = 1 / 511
        self.step1  = 1 / 511
        print ""
        print "Lena Plane"
        print "data", self.data
        print "ndim", self.ndim
        print "step", self.step0, self.step1
        
    def vtkActor(self):
        """
        Returns an object of type vtkLODActor for rendering within a VTK pipeline
        """
        import vtk
        me  = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        cts = vtk.vtkCellArray()
            
        for n in range(len(self.points)):
            pts.InsertPoint(n,self.points[n,0],self.points[n,1],self.points[n,2])
            
        for n in range(len(self.connectivity)):
            cts.InsertNextCell(3)
            for node in self.connectivity[n]:
                cts.InsertCellPoint(node)
              
        me.SetPoints(pts)
        me.SetPolys(cts)
              
        jpeg = vtk.vtkJPEGReader()
        jpeg.SetFileName(self.texture)
        textureCoordinates = vtk.vtkFloatArray()
        textureCoordinates.SetNumberOfComponents(3)
        textureCoordinates.SetName("TextureCoordinates")
        
        notablepoints = [[0,0,0],[1,0,0],[1,1,0],[0,1,0]]
        
        for p in notablepoints:
            textureCoordinates.InsertNextTuple(p)
            
        me.GetPointData().SetTCoords(textureCoordinates)
        
        texture = vtk.vtkTexture()
        texture.SetInputConnection(jpeg.GetOutputPort())
                
              
        dataMapper = vtk.vtkPolyDataMapper()
        dataMapper.SetInput(me)

        dataActor =vtk.vtkActor()
        dataActor.SetMapper(dataMapper)
        dataActor.SetTexture(texture)
            
        return [dataActor]
       
if __name__=="__main__":
    """
    This unit test is not for the class mirror, but for the Object class, and it
    verifies if the rotation and translation operations are consistent
    """
    print "###     TESTING GEOMETRICAL OPERATIONS OF OBJECT CLASS            ###"
    theorypoints = np.array([[0,-1,-1],
                             [0,+1,-1],
                             [0,+1,+1],
                             [0,-1,+1]])*0.5
    m = Mirror()
    assert np.linalg.norm(m.points - theorypoints) < 1e-6
    
    print "testing translation"
    m.translate(np.array([0,1,1])*0.5)
    theorypoints = np.array([[0,-0,-0],
                             [0,+2,-0],
                             [0,+2,+2],
                             [0,-0,+2]])*0.5
    assert np.linalg.norm(m.points - theorypoints) < 1e-6
    
    print "testing rotation aroung axis passing through origin"
    m.translate(np.array([0,-1,-1])*0.5)
    m.rotateAroundAxis(90,np.array([1,0,0]))
    theorypoints = np.array([[0,+1,-1],
                             [0,+1,+1],
                             [0,-1,+1],
                             [0,-1,-1]])*0.5                          
    assert np.linalg.norm(m.points - theorypoints) < 1e-6       

    m.rotateAroundAxis(360,np.array([1,0,0]))
    assert np.linalg.norm(m.points - theorypoints) < 1e-6     
    
    m.rotateAroundAxis(-90,np.array([1,0,0]))
    theorypoints = np.array([[0,-1,-1],
                             [0,+1,-1],
                             [0,+1,+1],
                             [0,-1,+1]])*0.5
    assert np.linalg.norm(m.points - theorypoints) < 1e-6   

    m.rotateAroundAxis(90,np.array([0,1,0]))
    theorypoints = np.array([[-1,-1,-0],
                             [-1,+1,-0],
                             [+1,+1,+0],
                             [+1,-1,+0]])*0.5
    assert np.linalg.norm(m.points - theorypoints) < 1e-6  
    assert np.linalg.norm(m.x - np.array([0,0,-1])) < 1e-6  
    m.rotateAroundAxis(-90,np.array([0,1,0]))
    
    print "testing the align to axis function"
    m.alignTo(np.array([1,0,0]),np.array([0,1,0]))
    theorypoints = np.array([[0,-1,-1],
                             [0,+1,-1],
                             [0,+1,+1],
                             [0,-1,+1]])*0.5
    assert np.linalg.norm(m.points - theorypoints) < 1e-6   
    
    m.alignTo(np.array([-1,0,0]),np.array([0,1,0]))
    theorypoints = np.array([[0,-1,+1],
                             [0,+1,+1],
                             [0,+1,-1],
                             [0,-1,-1]])*0.5
    assert np.linalg.norm(m.points - theorypoints) < 1e-6   
    
    m.alignTo(np.array([1,0,0]),None,np.array([0,0,1]))
    theorypoints = np.array([[0,-1,-1],
                             [0,+1,-1],
                             [0,+1,+1],
                             [0,-1,+1]])*0.5
    assert np.linalg.norm(m.points - theorypoints) < 1e-6                                

    print "testing rotation by axis off-origin"
    m.rotateAroundAxis(180,np.array([1,0,0]),np.array([0,1,1])*0.5)
    theorypoints = np.array([[0,+1.5,+1.5],
                             [0,+0.5,+1.5],
                             [0,+0.5,+0.5],
                             [0,+1.5,+0.5]])
    assert np.linalg.norm(m.points - theorypoints) < 1e-6      

    t = Texture()
    t.rotateAroundAxis(12,t.y)
    print t.x, t.y, t.z
    import Utils
    Utils.displayScenario([t])
    
    