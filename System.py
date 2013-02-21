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
import threading
import numpy as np
"""
The following imports are not mandatory (e.g., the user might not have Python
VTK installed. So the try/except blocks are programmed accordingly
"""
try:
    import vtk
    VTK_PRESENT = True
except:
    VTK_PRESENT = False
    print "VTK not found"
    
try:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    import matplotlib.pyplot as plt
    MPL_PRESENT = True
except:
    MPL_PRESENT = False
    print "Matplotlib not found or not compatible"

class Visitor(object):
    def __init__(self):
        pass
    def visit(self, obj):
        raise NotImplementedError
        
class VTKPlotter(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.actorList = []
        
    def visit(self, obj):       
        # Will not plot something without points
        if obj.points is None:
            return None
        
        # If has no connectivity, it is probably a line
        if not hasattr(obj, 'connectivity'):
            self.actorList.append(self.lineActor(obj))
        else:
            self.actorList.append(self.polyActor(obj))
            
            
    def display(self,displayAxes=True):
        if displayAxes:
            axesActor = vtk.vtkAxesActor()
            axesActor.SetShaftTypeToLine()
            self.actorList.append(axesActor)
                    
        window = PIVSimWindow()
        window.addActors(self.actorList)
        window.start()
        return window

        
    def lineActor(self,obj):
        """
        Returns an object of type vtkLODActor for rendering within a VTK pipeline
        """
        me  = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        cts = vtk.vtkCellArray()
            
        for n in range(len(obj.points)):
            pts.InsertPoint(n,obj.points[n][0],obj.points[n][1],obj.points[n][2])
            
        for n in range(1,len(obj.points)):
            cts.InsertNextCell(2)
            cts.InsertCellPoint(n-1)
            cts.InsertCellPoint(n)
              
        me.SetPoints(pts)
        me.SetLines(cts)
                          
        dataMapper = vtk.vtkPolyDataMapper()
        dataMapper.SetInput(me)

        dataActor =vtk.vtkLODActor()
        dataActor.SetMapper(dataMapper)
        
        if obj.color is not None:
            dataActor.GetProperty().SetColor(obj.color[0],obj.color[1],obj.color[2])
        if obj.opacity is not None:
            dataActor.GetProperty().SetOpacity(obj.opacity)
            
        return dataActor
    
    def polyActor(self,obj): 
        actor   = vtk.vtkPolyData()
        pts     = vtk.vtkPoints()
        cts     = vtk.vtkCellArray()
            
        for n in range(len(obj.points)):
            pts.InsertPoint(n,obj.points[n,0],obj.points[n,1],obj.points[n,2])
           
        for n in range(len(obj.connectivity)):
            cts.InsertNextCell(3)
            for node in obj.connectivity[n]:
                cts.InsertCellPoint(node) 
                
        actor.SetPoints(pts)
        actor.SetPolys(cts)
        
        if obj.normals is not None:
            nrm = vtk.vtkDoubleArray()
            nrm.SetNumberOfComponents(3)
            nrm.SetNumberOfTuples(len(obj.points))
            for n in range(len(obj.points)):
                nrm.SetTuple(n,obj.normals[n].tolist())
            actor.GetPointData().SetNormals(nrm)
            
        dataMapper = vtk.vtkPolyDataMapper()
        dataMapper.SetInput(actor)
        
        dataActor =vtk.vtkLODActor()
        dataActor.SetMapper(dataMapper)
        
        if obj.color is not None:
            dataActor.GetProperty().SetColor(obj.color[0],obj.color[1],obj.color[2])
        if obj.opacity is not None:
            dataActor.GetProperty().SetOpacity(obj.opacity)
            
        return dataActor
    
class PythonPlotter(Visitor):
    def __init__(self):
        Visitor.__init__(self)
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('PyVSim Visualization window ' + \
                                          '- powered by Matplotlib') 
        self.ax  = self.fig.gca(projection='3d')
        
    def visit(self, obj):       
        # Will not plot something without points
        if obj.points is None:
            return None
        
        # If has no connectivity, it is probably a line
        if not hasattr(obj, 'connectivity'):
            self.lineActor(obj)
        else:
            self.polyActor(obj)
            
            
    def display(self,displayAxes=True):
        plt.show()
 
    def lineActor(self,obj):
        col = Line3DCollection([obj.points])      
        if obj.color is not None: 
            if obj.opacity is None:
                obj.opacity = 1
            col.set_color([obj.color[0],
                           obj.color[1],
                           obj.color[2],
                           obj.opacity])
        # color mapping
        #col.set_array(val)
        #col.set_cmap(cm.hot)
        self.ax.add_collection(col)

    
    def polyActor(self,obj): 
        for n in range(np.size(obj.connectivity,0)):
            col = Poly3DCollection([obj.points[obj.connectivity]])
            col = Poly3DCollection([obj.points[obj.connectivity[n]]])
            
            if obj.color is None: 
                color = [0.5,0.5,0.5]
            else:
                color = obj.color
                
            if obj.opacity is None:
                opacity = 0.3
            else:
                opacity = obj.opacity
                
            col.set_color([color[0],
                           color[1],
                           color[2],
                           opacity])
            
            col.set_edgecolor([color[0],
                               color[1],
                               color[2],
                               opacity])
            
            # color mapping
            #col.set_array(val)
            #col.set_cmap(cm.hot)
            self.ax.add_collection(col)

def Plotter(mode="vtk"):
    """
    Returns a plotter visitor, emulating a factory class.
    This uses information about the python installation to return a 
    compatible plotter.
    
    mode can be either "vtk" or "mpl"
    
    "vtk"
        Uses the Python wrapping of the Visualization Toolkit (VTK) to plot
        the environment
        
    "mpl"
        Uses Matplotlib to plot the environment
    """
    if mode == "vtk" and VTK_PRESENT:
        return VTKPlotter()
    
    if mode == "vtk" and not VTK_PRESENT and MPL_PRESENT:
        print "Could not return a VTK plotter, returning a Matplotlib plotter"
        return PythonPlotter()
    
    if mode == "mpl" and MPL_PRESENT:
        return PythonPlotter()
    
    if mode == "mpl" and VTK_PRESENT and not MPL_PRESENT:
        print "Could not return a Matplotlib plotter, returning a VTK plotter"
        return VTKPlotter()
    
    raise ImportError("Could not import a library for plotting. PyVSim" + \
                        "uses both Matplotlib and VTK")

    
class PIVSimWindow(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)        
        self.footText       = "PyVSim ver. 1.0"
        self.windowTitle    = "PyVSim Visualization window - powered by VTK"
        self.windowColor    = [0,0.35,0.55]
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