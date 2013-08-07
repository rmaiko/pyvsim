"""
.. module :: System
    :platform: Unix, Windows
    :synopsis: Base classes for making pyvsim work
    
This module contains the methods and classes used for interfacing "with the
System". Basically, the two functionalities of this module are:

* File handling
* Displaying
    
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
import threading
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import Core
import json
import re
import cPickle
try:
    import vtk
    VTK_PRESENT = True
except ImportError:
    VTK_PRESENT = False
    print "VTK not found"
    
VERSION = "1.0"

def plot(obj, mode="vtk", displayAxes = True):
    """
    Convenience function to avoid the routine of creating visitor, making it
    visit the part tree and ask for plotting.
    
    *Warning* - Due to bugs both in Matplotlib and VTK, the program execution
    is paused until the window is closed. So plotting should be the last 
    operation in the pipeline.
    
    Parameters
    ----------
    obj : Primitives.Assembly
        The scenario to be displayed
    mode : "vtk" or "mpl"
        The library to be used in displaying the scenario, the standard option
        is "vtk", which is faster and supports rendering complex scenarios.
        Some scenarios, however, can be plotted using Matplotlib.
    displayAxes : boolean
        When plotting with VTK, the axis can be hidden by setting this option
        to false.
    """
    plotter = Plotter(mode)
    obj.acceptVisitor(plotter)
    plotter.display(displayAxes)
    
def save(obj, filename=None, mode="pickle"): 
    """
    Convenience function for saving a Primitives.Assembly object.
    
    Parameters
    ----------
    obj : Primitives.Assembly
        The scenario to be saved. 
    filename : string
        The name of the output file
    mode : "pickle" or "json"
        Specifies the file format to be used in saving. The default option is
        pickle (which is fast and bug free), use JSON only if you need to edit
        the file in a human-readable way.
        
    """
    if mode == "pickle":
        saver = Saver(mode = Saver.PICKLE)
    elif mode == "json":
        saver = Saver(mode = Saver.JSON)
        
    obj.acceptVisitor(saver)
    saver.dump(filename)

def load(filename, mode = None):
    """
    Convenience function for loading a part tree. Pickle/JSON is chosen
    by trial and error
    
    Parameters
    ----------
    filename : string
        The name of the input file
    mode : "pickle" or "json"
        Specifies the file format of the input file. The default option is
        None, which will make the loader first try to unpickle the file
        (which is fast and bug free). Use JSON only if you need to edit
        the file in a human-readable way.
        
    Raises
    ------
    ValueError
        If decoding was not possible
    """       
    f = open(filename,'r')
    try:
        rawdata = cPickle.load(f)
        return rawdata
    except cPickle.UnpicklingError:
        print "Could not decode pickle, trying JSON"
    finally:
        f.close()
        
    f = open(filename,'r')
    try:
        rawdata = json.load(f, cls = pyvsimJSONDecoder)
    finally:
        f.close()
        
    return rawdata


        
        

def Plotter(mode="vtk"):
    """
    This is a factory that returns a plotter visitor.
    
    It uses information about the Python installation to return a 
    compatible plotter.
    
    Parameters
    ----------
    mode : "vtk" or "mpl"
        "vtk" Uses the Python wrapping of the Visualization Toolkit (VTK) to 
        plot the environment, whereas "mpl" uses Matplotlib to plot the 
        environment (this option is not so efficient in plotting environments
        with complex objects)
    """
    if mode == "vtk":
        if VTK_PRESENT:
            return VTKPlotter()
        else:
            print "Could not import VTK, returning a Matplotlib plotter"
            return PythonPlotter()
    elif mode == "mpl":
        return PythonPlotter()
    else:
        raise ValueError("Could not understand input " + mode)
    
    raise ImportError("Could not import a library for plotting. PyVSim" +
                        "uses both Matplotlib and VTK")

class Visitor(object):
    """
    This is the prototype of a class capable of traversing the parts tree while
    having access to all objects, one at a time.
    """
    def __init__(self):
        pass
    def visit(self, obj):
        raise NotImplementedError
        
class VTKPlotter(Visitor):
    """
    This class is a Facade to VTK. It takes a snapshot of the assembly tree 
    and generates a VTK plot.
    """
    def __init__(self):
        Visitor.__init__(self)
        self.actorList = []
        
    def visit(self, obj):  
        if obj.PLOTDIMS == -1:
            return None
        elif obj.PLOTDIMS == 0:
            self.actorList.append(self.pointsActor(obj))
        elif obj.PLOTDIMS == 1:
            self.actorList.append(self.lineActor(obj))
        elif obj.PLOTDIMS == 3:
            self.actorList.append(self.polyActor(obj))
        else:
            raise ValueError("Attempted to plot a non pyvsim object")                 
            
    def display(self,displayAxes=True):
        """
        Creates a window and plots all the objects found during the last visit
        to the parts tree.
        
        I.e., before running this command, you should do::
        
        main_assembly.acceptVisitor(plotter)
        
        Attention: This will stop the program execution until the
        window is closed. This is a "feature" of matplotlib and VTK.
        """
#        print "There are %i elements to be plotted" % len(self.actorList)                    
        window = self.VTKWindow(displayAxes)
        window.addActors(self.actorList)
        window.start()
        return window

    def pointsActor(self, obj):
        npts = np.size(obj.points, 0)
         
        vertices = vtk.vtkCellArray()
        ptsource = vtk.vtkPoints()
        
        ptsource.SetNumberOfPoints(npts)
        
        for n,p in enumerate(obj.points):
            ptsource.SetPoint(n,p)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(n)
            
        point = vtk.vtkPolyData()
        point.SetPoints(ptsource)
        point.SetVerts(vertices)   
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(point) 
        
        actor = vtk.vtkLODActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(1)
        return actor
        
    def lineActor(self,obj):
        """
        Returns an object of type vtkLODActor for rendering within a VTK 
        pipeline
        """
        me  = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        cts = vtk.vtkCellArray()
            
        for n in range(len(obj.points)):
            pts.InsertPoint(n,obj.points[n][0],
                              obj.points[n][1],
                              obj.points[n][2])
            
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
            if np.size(obj.color) == 3:
                dataActor.GetProperty().SetColor(obj.color[0],
                                                 obj.color[1],
                                                 obj.color[2])
            else:
                carray = vtk.vtkUnsignedCharArray()
                carray.SetNumberOfComponents(3)
                carray.SetName("Colors")
                color = (obj.color*255).astype(int)
                for c in color:
                    carray.InsertNextTupleValue(c)
                me.GetCellData().SetScalars(carray)

        if obj.opacity is not None:
            dataActor.GetProperty().SetOpacity(obj.opacity)
            
        if obj.width is not None:
            dataActor.GetProperty().SetLineWidth(obj.width)
            
        return dataActor
    
    def polyActor(self,obj): 
        """
        Returns an object of type vtkLODActor for rendering within a VTK 
        pipeline
        """
        actor   = vtk.vtkPolyData()
        pts     = vtk.vtkPoints()
        cts     = vtk.vtkCellArray()
            
        for n in range(len(obj.points)):
            pts.InsertPoint(n,obj.points[n,0], 
                              obj.points[n,1],
                              obj.points[n,2])
           
        for n in range(len(obj.connectivity)):
            cts.InsertNextCell(3)
            for node in obj.connectivity[n]:
                cts.InsertCellPoint(node) 
                
        actor.SetPoints(pts)
        actor.SetPolys(cts)
        
        # If the normals of the object are specified (smooth object), this is
        # rendered as such
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
            dataActor.GetProperty().SetColor(obj.color[0],
                                             obj.color[1],
                                             obj.color[2])
        if obj.opacity is not None:
            dataActor.GetProperty().SetOpacity(obj.opacity)
            
        return dataActor
    
    class VTKWindow(threading.Thread):
        """
        A window to plot objects using VTK. Even though this stays in a separate
        thread, a error in VTK interface to Python makes it wait until the
        window is closed.
        """
        def __init__(self, displayAxes = True):
            threading.Thread.__init__(self)        
            self.footText       = "PyVSim ver. " + VERSION
            self.windowTitle    = "PyVSim Visualization window - powered by VTK"
            self.windowColor    = [0,0.25,0.40]
            self.windowSize     = [800,800]
            self.actorlist      = None
            self.displayAxes    = displayAxes
            
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
            if self.displayAxes:
                axesActor = vtk.vtkAxesActor()
                axesActor.SetShaftTypeToLine()
                axes = vtk.vtkOrientationMarkerWidget()
                axes.SetOrientationMarker(axesActor);
                axes.SetInteractor(self.interactor);
                axes.EnabledOn();
                axes.InteractiveOn();
            self.window.Render()
            self.window.SetWindowName(self.windowTitle)
            self.interactor.Start()
    
class PythonPlotter(Visitor):
    """
    This visitor class creates a plot of the assembly tree using the
    Matplotlib library from python.
    """
    def __init__(self):
        Visitor.__init__(self)
        self.fig = plt.figure()
        self.fig.canvas.set_window_title('PyVSim Visualization window ' + 
                                         'ver. ' + VERSION + 
                                          ' - powered by Matplotlib') 
        self.ax  = self.fig.gca(projection='3d')
        
    def visit(self, obj):  
        """
        Takes a snapshot of the object and creates a elements in a Matplotlib
        window.
        """     
        if obj.PLOTDIMS == -1:
            return None
        elif obj.PLOTDIMS == 0:
            self.pointsActor(obj)
        elif obj.PLOTDIMS == 1:
            self.lineActor(obj)
        elif obj.PLOTDIMS == 3:
            self.polyActor(obj)
        else:
            raise ValueError("Attempted to plot a non pyvsim object")              
            
    def display(self,displayAxes=True):
        """
        Creates a window and plots all the objects found during the last visit
        to the parts tree.
        
        I.e., before running this command, you should do::
        
        main_assembly.acceptVisitor(plotter)
        
        Attention: This will stop the program execution until the
        window is closed. This is a "feature" of matplotlib and VTK.
        """
        _ = displayAxes
        plt.show()
        
    def pointsActor(self, obj):
        self.ax.scatter3D(obj.points[:,0],obj.points[:,1],obj.points[:,2])
 
    def lineActor(self,obj):
        """
        Adds a collection to the current axes to draw a line
        """
        col = Line3DCollection([obj.points])      
        if obj.color is not None: 
            if obj.opacity is None:
                obj.opacity = 1
            col.set_color([obj.color[0],
                           obj.color[1],
                           obj.color[2],
                           obj.opacity])
        self.ax.add_collection(col)
    
    def polyActor(self,obj): 
        """
        Adds a collection to the current axes to draw a surface
        """
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

class Saver(Visitor):
    """
    This is the standard PyVSim saving routine, using python cPickle. The
    performance is quite good and does not require complicated parsing and
    conversion of the tree. However, the result is not human readable.
    
    Use when performance and reliability are desired.
    """
    JSON   = 1
    PICKLE = 0
    def __init__(self, mode = JSON):
        self.topickle = None
        self.mode     = mode
        
    def visit(self, obj):
        if obj.parent is None:
            self.topickle = obj
            
    def dump(self, name = None):
        if self.mode == Saver.PICKLE:
            if name is None:
                cPickle.dumps(self.topickle)
            else:
                f = open(name,'w')
                try:
                    cPickle.dump(self.topickle, f)
                finally:
                    f.close()
                    
        elif self.mode == Saver.JSON:
            if name == None:
                json.dumps(self.topickle, 
                           cls = pyvsimJSONEncoder,
                           indent = 2)
            else:
                f = open(name,'w')
                try:
                    json.dump(self.topickle, f, 
                          cls = pyvsimJSONEncoder, 
                          indent = 2)
                finally:
                    f.close()
        

class pyvsimJSONEncoder(json.JSONEncoder):
    """
    A JSON Encoder capable of dealing with a pyvsim simulation tree without
    creating duplicates of objects and solving circular references (at least
    the type found naturally in a pyvsim tree).
    
    This class is also aware of PyvsimObjects and numpy.ndarrays.
    """
    FILEMODE   = None
    SCREENMODE = 2 
    def __init__(self,
                 skipkeys          = False, 
                 ensure_ascii      = True,
                 check_circular    = False, 
                 allow_nan         = True, 
                 sort_keys         = True,
                 indent            = None, 
                 separators        = None, 
                 encoding          = 'utf-8', 
                 default           = None):
        json.JSONEncoder.__init__(self, 
                                  skipkeys          = False, 
                                  ensure_ascii      = True,
                                  check_circular    = False, 
                                  allow_nan         = True, 
                                  sort_keys         = sort_keys,
                                  indent            = indent, 
                                  separators        = None, 
                                  encoding          = 'utf-8', 
                                  default           = None)
        self.serializedObjects  = {}
        
    def default(self, obj):      
        """
        Objects are encoded in a special dictionary, with the magic key
        "object_type", which is essential for the unpacking of the object.
        
        It exploits the property __dict__ of the objects to pack the data 
        """
        if isinstance(obj, object):
            temp = {}
            temp["object_module"] = str(obj.__class__.__module__)
            temp["object_type"]   = str(obj.__class__.__name__)
            temp["object_id"]   = id(obj)
            if isinstance(obj, np.ndarray):
                temp["dtype"]       = str(obj.dtype)
                temp["data"]        = obj.tolist()
                return temp
            if self.serializedObjects.has_key(id(obj)):
                temp["object_dict"] = None
            else: 
                try:   
                    temp["object_dict"] = obj.__getstate__()
                except AttributeError:
                    temp["object_dict"] = obj.__dict__

                self.serializedObjects[id(obj)] = temp
            return temp   
        
        return json.JSONEncoder.default(self, obj)

class pyvsimJSONDecoder(json.JSONDecoder):
    """
    Extension of the python JSONDecoder class aware of pyvsim objects (via the
    PyvsimObject interface) and numpy.ndarrays
    """
    def __init__(self, 
                 encoding=None, 
                 object_hook=None, 
                 parse_float=None,
                 parse_int=None, 
                 parse_constant=None, 
                 strict=True,
                 object_pairs_hook=None):
        json.JSONDecoder.__init__(self, object_hook=self.default, strict = False)
        self.cornFlakes = {}
        
    def default(self, obj): 
        """
        Objects are encoded in a special-purpose dictionary with the magic
        entry "object_type". When this key is found, an object of the type is
        instantiated and receives the data contained in the dictionary.
        
        Another problem is dealing with object copies. The variable cornFlakes
        stores a reference to each object that was already unpacked, and when
        the internal identifiers match, the pointer is reused.
        """
        if (isinstance(obj, Core.PyvsimObject) or 
            obj is None or 
            isinstance(obj, np.ndarray)):
            return obj
                        
        if obj.has_key("object_type"):
            if not self.cornFlakes.has_key(obj["object_id"]):
                if obj.has_key("dtype"):
                    obj["data"] = self.treatArray(obj)
                    myobject = obj["data"]
                else:
                    p = re.split("[\.\']",obj["object_module"])
                    pkg         = __import__(p[0])
                    mod         = getattr(pkg,p[1])
                    myobject    = getattr(mod,obj["object_type"])()
                self.cornFlakes[obj["object_id"]] = myobject
                
            try:
                if obj["object_dict"] is not None:
                    self.cornFlakes[obj["object_id"]].__dict__ = obj["object_dict"]
            except KeyError:
                pass
                
            return self.cornFlakes[obj["object_id"]]  
       
        return obj
    
    def treatArray(self, obj):
        """
        This is a tricky method when the numpy.ndarray contains objects (for
        example in the list of ray tracing intersections). 
        """
        if obj["dtype"] == "object":
            obj["data"] = np.array(obj["data"])
            iterator    = np.nditer(obj["data"],
                                    flags=['refs_ok','multi_index'])
            
            while not iterator.finished:
                obj["data"][iterator.multi_index] = \
                            self.default(obj["data"][iterator.multi_index])
                iterator.iternext()
        return np.array(obj["data"])