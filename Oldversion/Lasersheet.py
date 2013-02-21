from __future__ import division
import vtk
import numpy as np
import copy
import Object
import Ray
import math
import Utils
import vec
import LaserProfiles

class Lasersheet(Object.Object):
    def __init__(self):
        # Properties inherited from Object
        Object.Object.__init__(self)
        self.indexOfRefraction          = 0
        self.indexOfRefractionAmbient   = 0
        self.indexOfReflection          = 0
        self.isLightSource              = 1
        self.points                     = np.array([])
        self.connectivity               = np.array([])
        self.normals                    = None
        self.color                      = [0,180,0]
        self.opacity                    = 0.8
        # Properties specific to the class        
        self.sideVectors                = None
        self.boundaryVectors            = None
        self.hexaPoints                 = np.array([])
        self.hexaConnectivity           = np.array([])
        # Geometrical properties
        self.planeDivergence            = 15
        self.thicknessDivergence        = 1
        self.intensityFunction          = LaserProfiles.gaussianIntensity
        self.intensityMultiplier        = 3e3   # J / m^2
        self.referenceArea              = 0.02 * 0.005
        # Property used for tracing
        self.usefulLength               = 5
        self.initialOffset              = 0.01
        self.wavelength                 = 532e-9
        # Safety calculation parameters
        self.maximumRayTrace            = 100
        self.raysInPlane                = 30
        self.raysInThickness            = 3
        self.reflectedRays              = []
        self.calculateVectors()
                
        # Parameter for plotting (display reflections or only useful length)
        self.displayReflections         = False  
    
    def traceReflections(self,environment):
        """
        This method traces the lightsheet path, considering that all non-optical
        objects are reflective
        """
        for w1 in np.linspace(1,0,self.raysInPlane):
            for w2 in np.linspace(1,0,self.raysInThickness):
                v = Utils.EQBInterpolation(np.array([w1,w2,0]),
                                                np.array([0,0,0]),
                                                np.array([1,0,0]),
                                                np.array([1,1,0]),
                                                np.array([0,1,0]),
                                                self.boundaryVectors[0],
                                                self.boundaryVectors[1],
                                                self.boundaryVectors[2],
                                                self.boundaryVectors[3])
                v = v / np.linalg.norm(v)
                self.reflectedRays.insert(0,Ray.Ray(self.origin,v))
                self.reflectedRays[0].bounceOnAllObjects     = True
                self.reflectedRays[0].maximumRayTrace        = self.maximumRayTrace
                self.reflectedRays[0].stepRayTrace           = self.maximumRayTrace
                self.reflectedRays[0].trace(environment)
        
    def trace(self,environment):
        """
        This method traces the lightsheet path for the creation of elements to
        be used in the PIV simulation. 
        
        In order to save memory, please set the parameter usefulLength 
        to a low value (only what is needed for PIV calculation)
        
        In order to execute raytracing for safety purposes, refer to the 
        function traceReflections
        """
        # Tracing of the main plane
        r = []
        for n in range(2):
            r.insert(0,Ray.Ray(self.origin,self.sideVectors[n]))
            r[0].bounceOnAllObjects     = False
            r[0].maximumRayTrace        = self.usefulLength
            r[0].stepRayTrace           = self.usefulLength
            r[0].trace(environment)
        pts = [self.origin,r[1].points[1],r[0].points[1]]
        cts = [[0,1,2]]
        for n in range(2,len(r[0].points)):
            pts.append(r[1].points[n])
            pts.append(r[0].points[n])
            cts.append([2*n  , 2*n-3 , 2*n-1])
            cts.append([2*n  , 2*n-2 , 2*n-3])
            
        self.points         = np.array(pts)
        self.connectivity   = np.array(cts)
            
        # Tracing of the bounding volumes
        r       = []
        pts     = []
        cts     = []
        for n in range(4):
            # not very elegant way to solve problem that first hexa would be degenerate
            initialPoint = self.origin + self.initialOffset*self.boundaryVectors[n]
            r.insert(0,Ray.Ray(initialPoint,self.boundaryVectors[n]))
            r[0].bounceOnAllObjects     = False  # Avoid bumping at dumps
            r[0].maximumRayTrace        = self.usefulLength
            r[0].stepRayTrace           = self.usefulLength # reduce nb of hexas
            r[0].trace(environment) 
        
        # Determine the number of points - if rays follow too different paths,
        # it will truncate while geometry is still valid
        minpoints = 9999
        for rn in range(4):
            if len(r[rn].points) < minpoints:
                minpoints = len(r[rn].points)
        
        # Define hexas
        for n in range(minpoints):
            for rn in range(4):
                pts.append(r[rn].points[n])
            if (n > 0):                                 # Definition as in VTK :
                cts.append(range((n+1)*4-8,(n+1)*4))    # (0,1,2,3) is the base of the hexahedron which, 
        self.hexaPoints         = np.array(pts)         # using the right hand rule, forms a quadrilaterial 
        self.hexaConnectivity   = np.array(cts)         # whose normal points in the direction of the 
                                                        # opposite face (4,5,6,7).  
    def calculateIntensity(self,p):
        """
        This method receives a list of three-dimensional points and calculate the
        light intensity and direction that the lasersheet illuminates each.
        
        Unhappilly, for a lasersheet that spans a point more than once, only the 
        higher magnitude illumination is considered, this is because for Mie
        scattering calculations, considering two sources is very hard.
        
        The function receives a (n x 3) numpy array and returns a (n x 3) array
        with vectors which magnitude correspond to that of the illumination, in 
        J/m^2
        """
        if p.ndim > 1:
            I = np.zeros((len(p),3))
            Iscalars = np.zeros((len(p)))
        else:
            I = np.zeros((3))
            Iscalars = 0
            
        for hexa in self.hexaConnectivity:
            inHexa = Utils.pointInHexa(p,self.hexaPoints[hexa])
            # print "inHexa ", inHexa
            p1 = np.sum(self.hexaPoints[hexa[4:8]],0) / 4
            p0 = np.sum(self.hexaPoints[hexa[0:4]],0) / 4
            N = vec.normalize(p1 - p0)
            intersectionParameters = vec.dot(N,p-p0)
            planePoint = p0 + vec.listTimesVec(vec.dot(N,p-p0),N)
            
            sides = [0,0,0,0]
            Pside = [0,0,0,0]
            for n in range(4):
                sides[n] =(self.hexaPoints[hexa[n+4]]-self.hexaPoints[hexa[n]])
                Pside[n] = self.hexaPoints[hexa[n]] + \
                            vec.listTimesVec(vec.dot(N,planePoint-self.hexaPoints[hexa[n]]),
                                                     (sides[n] / vec.dot(N,sides[n])))
                sides[n] = vec.normalize(sides[n])

            V = Utils.EQBInterpolation(p,Pside[0],Pside[1],Pside[2],Pside[3], \
                                         sides[0],sides[1],sides[2],sides[3])
            V = vec.normalize(V)
            
            Area = Utils.quadArea(Pside[0],Pside[1],Pside[2],Pside[3])
            

            param = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
            P = Utils.EQBInterpolation(p,Pside[0],Pside[1],Pside[2],Pside[3], \
                                         param[0],param[1],param[2],param[3])  

            # Calculate the magnitude of the illumination each particle receives
            Iscalarstemp = inHexa* \
                           self.intensityFunction(P[:,0],P[:,1])* \
                           self.intensityMultiplier * \
                           self.referenceArea/Area
            # print "Iscalarstemp ", Iscalarstemp
            # Substitute the values in Iscalars if a higher magnitude is found
            # print "Iscalars before ", Iscalars
            Iscalars_old = Iscalars
            Iscalars = Iscalars     * (Iscalarstemp <= Iscalars) + \
                       Iscalarstemp * (Iscalarstemp > Iscalars)   
            # print "Iscalars after ", Iscalars
            # Calculate the illumination vectors (direction + itnensity)
            Itemp = vec.listTimesVec(Iscalarstemp,V)
            # print "Itemp ", Itemp
            I = vec.listTimesVec((Iscalarstemp <= Iscalars_old),I) + \
                vec.listTimesVec((Iscalarstemp >  Iscalars_old),Itemp)  
            # print "I ", I
        return I   
                                         
    def setDivergences(self,planeDivergence,thicknessDivergence):
        """
        This is the method for setting the spreads of the laser sheet, do not set
        them manually, as this can lead to inconsistent calculations and plotting
        Inputs:
        planeDivergence (degrees)      - in-plane divergence of laser plane
        thicknessDivergence (degrees)  - out of plane divergence of laser plane
        """
        self.planeDivergence        = planeDivergence
        self.thicknessDivergence    = thicknessDivergence
        self.destroyData()
        
    def destroyData(self):
        """
        This method erases data that is calculated from the lasersheet interaction
        with its ambient. 
        This must be called whenever there is a change in geometrical properties
        of the sheet, otherwise this can lead to incorrect calculations and plotting
        """
        self.calculateVectors()
        self.points             = np.array([])
        self.connectivity       = np.array([])
        self.hexaPoints         = np.array([])
        self.hexaConnectivity   = np.array([])
        self.reflectedRays      = []
        
    def calculateVectors(self):
        """
        This routine is used to calculate the side and boundary vectors of the
        lasersheet.
        
        Side vectors are used to trace the lasersheet main plane
        
        Boundary vectors are used to define the volume illuminated by the sheet
        
        This routine will destroy data for creation of a visual representation of
        the lasersheet, because a change in the lasersheet direction requires a 
        change in all the raytracing procedure.
        
        """
        self.mainVector         = self.x
        
        self.sideVectors        = [Utils.rotateVector(self.mainVector,-self.planeDivergence/2,self.y),
                                   Utils.rotateVector(self.mainVector,+self.planeDivergence/2,self.y)]

        self.boundaryVectors    = [Utils.rotateVector(self.sideVectors[0],+self.thicknessDivergence/2,self.z),
                                   Utils.rotateVector(self.sideVectors[1],+self.thicknessDivergence/2,self.z),
                                   Utils.rotateVector(self.sideVectors[1],-self.thicknessDivergence/2,self.z),
                                   Utils.rotateVector(self.sideVectors[0],-self.thicknessDivergence/2,self.z)]
                                          
    def vtkActor(self):
        """
        Returns a list of objects of type vtkLODActor for rendering within a VTK pipeline
        """
        me  = []
        
        # Plotting of the safety reflections
        if self.displayReflections:
            for r in self.reflectedRays:
                me.insert(0,r.vtkActor()[0])    
        
        # Plotting of the main plane
        me.insert(0,Object.Object.vtkActor(self)[0])
        
        # Plotting of hexas (must do manually)
        hexas   = vtk.vtkUnstructuredGrid()
        pts     = vtk.vtkPoints()
        cts     = vtk.vtkCellArray()
            
        for n in range(len(self.hexaPoints)):
            pts.InsertPoint(n,self.hexaPoints[n,0],self.hexaPoints[n,1],self.hexaPoints[n,2])
            
        for n in range(len(self.hexaConnectivity)):
            cts.InsertNextCell(8)
            for node in self.hexaConnectivity[n]:
                cts.InsertCellPoint(node)
                
        hexas.SetPoints(pts)
        hexas.SetCells(12,cts) #Incredibly, the magic number for hexa in VTK is 12
        hexaMapper = vtk.vtkDataSetMapper()
        hexaMapper.SetInput(hexas)
        hexaActor = vtk.vtkActor()
        hexaActor.SetMapper(hexaMapper)
        hexaActor.GetProperty().SetOpacity(0.6)
        me.insert(0,hexaActor)
                
        return me        
               
if __name__=="__main__":
    """
    Code for unit testing ray class
    """
    
    import vtk
    import Utils
    import pprint
    import Planes
    import time
    
    # Mirror 1
    mesh = Planes.Mirror(1,1)
    mesh.translate(np.array([1,0,0]))
    mesh.rotateAroundAxis(15.5,np.array([0,0,1]))
    
    # Mirror 2
    mesh2 = Planes.Mirror(5,5)
    mesh2.translate(np.array([-7,0,0]))
    
    # Lasersheet
    l = Lasersheet()
    l.thicknessDivergence = 10
    l.alignTo(np.array([1,0.5,0]),np.array([0,1,0]))
    l.translate([0,0,0])
    l.trace([mesh,mesh2])
    l.maximumRayTrace = 40
    l.traceReflections([mesh,mesh2])

    l.displayReflections         = True
    
    window = Utils.displayScenario([mesh,mesh2,l])