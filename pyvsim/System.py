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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import copy
import Primitives
import Core
import pprint 
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
        saver = Saver()
    elif mode == "json":
        saver = JSONSaver()
        
    obj.acceptVisitor(saver)
    saver.dump(filename)
    # cPickle.UnpicklingError
    # ValueError

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
    if mode is not "json":
        try:
            return Loader(filename)
        except cPickle.UnpicklingError:
            print "Could not decode pickle, trying JSON"
        
    return JSONLoader(filename)

        
        

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
    This class takes a snapshot of the assembly tree and generates a VTK plot.
    """
    def __init__(self):
        Visitor.__init__(self)
        self.actorList = []
        
    def visit(self, obj):       
        # Will not plot something without points
        if not hasattr(obj, 'points'):
            return None
        elif obj.points is None:
            return None
        
        # If has no connectivity, it is probably a line
        if not hasattr(obj, 'connectivity'):
            self.actorList.append(self.lineActor(obj))
        else:
            self.actorList.append(self.polyActor(obj))
            
            
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
        if displayAxes:
            axesActor = vtk.vtkAxesActor()
            axesActor.SetShaftTypeToLine()
            self.actorList.append(axesActor)
                    
        window = self.VTKWindow()
        window.addActors(self.actorList)
        window.start()
        return window

        
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
            dataActor.GetProperty().SetColor(obj.color[0],
                                             obj.color[1],
                                             obj.color[2])
        if obj.opacity is not None:
            dataActor.GetProperty().SetOpacity(obj.opacity)
            
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
        def __init__(self):
            threading.Thread.__init__(self)        
            self.footText       = "PyVSim ver. " + VERSION
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
        # Will not plot something without points
        if not hasattr(obj, 'points'):
            return None
        elif obj.points is None:
            return None
        
        # If has no connectivity, it is probably a line
        if not hasattr(obj, 'connectivity'):
            self.lineActor(obj)
        else:
            self.polyActor(obj)
            
            
    def display(self,displayAxes=True):
        """
        Creates a window and plots all the objects found during the last visit
        to the parts tree.
        
        I.e., before running this command, you should do::
        
        main_assembly.acceptVisitor(plotter)
        
        Attention: This will stop the program execution until the
        window is closed. This is a "feature" of matplotlib and VTK.
        """
        plt.show()
 
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
    def __init__(self):
        self.topickle = None
        
    def visit(self, obj):
        if obj.parent is None:
            self.topickle = obj
            
    def dump(self, name = None):
        if name is None:
            cPickle.dump(self.topickle)
        else:
            f = open(name,'w')
            try:
                cPickle.dump(self.topickle, f)
            finally:
                f.close()
           
def Loader(name):
    """
    Loads an assembly from a python pickle file. This is the preferred 
    implementation, as it's faster and more reliable.
    """
    f = open(name,'r')
    try:
        rawdata = cPickle.load(f)
    finally:
        f.close()
    return rawdata

class pyvsimJSONEncoder(json.JSONEncoder):
    """
    A JSON Encoder capable of dealing with a pyvsim simulation tree without
    creating duplicates of objects.
    """
    def default(self, obj):
        saneobject = obj 

        if isinstance(obj, Core.PyvsimObject):
            return obj.sanedict()
            
        return json.JSONEncoder.default(self, saneobject)


class JSONSaver(Visitor):
    """
    This class follows the visitor pattern to traverse the assembly tree
    and serialize all objects to JSON. There is quite a lot of conversion
    work being done, since there are many cross-references and numpy arrays
    (that are not supported by python json implementation), so this method
    is not fast and not very reliable. The advantage is that is generates
    human-readable (or almost) outputs.
    """
           
    def __init__(self):
        Visitor.__init__(self)
        self.myobjects = {}
        self.jsonEncoder = None
        
    def visit(self, obj):
        """
        This is the main method to access the assembly tree. Notice that
        this takes only a snapshot and doesn't store references, so if changes
        are made, the tree must be visited again.
        """
        self.myobjects[obj.__repr__()] = obj
        for key in obj.__dict__:
            inspected = obj.__dict__[key]
            if isinstance(inspected, Core.PyvsimObject):
                self.myobjects[inspected.__repr__()] = inspected
                
    def dump(self, name = None):
        """
        Use this to dump the snapshot taken with the "visit" method
        to a file or to the screen.
        
        As the intention of this class is to generate human-readable output,
        the file contains line breaks and indents.
        """
        if name is None:
            self.jsonEncoder = pyvsimJSONEncoder(indent = 4)
            pprint.pprint(self.jsonEncoder.encode(self.myobjects))
        else:
            f = open(name,'w')
            try:
                json.dump(self.myobjects, f, cls = pyvsimJSONEncoder, indent = 2)
            finally:
                f.close()

def JSONLoader(name):
    """
    This function returns an assembly tree with the contents of the specified
    file. Please note that absolutely no checks are performed to guarantee that
    the file is really a JSON (not a pickle).
   
    This implementation is definetely not elegant, as it has to check for some
    very specific data structures (viz. lists of lists of lists of strings) and
    reconstructs only simulation objects, however this seems to be the only
    simple way of doing JSON parsing of such objects, as json, contrary to
    pickle has no support for user classes.
    """
    f = open(name, 'r')
    try:
        filecontent = f.read()
    finally:
        f.close()
       
    allobjects = json.loads(filecontent)

    # Reconstruct all objects first
    objectlist = []
    idlist     = []
    for key in allobjects.keys():
        objectlist.append(_instantiateFromObjectString(key))
        idlist.append(allobjects[key]["_id"])
        objectlist[-1].__dict__ = allobjects[key]
       
    # Now reconstruct internal object references and numpy arrays
    for obj in objectlist:
        for key in obj.__dict__:
            # Reconstruct all lists to numpy arrays
            if type(obj.__dict__[key]) == list:
                obj.__dict__[key] = np.array(obj.__dict__[key])
                # Reconstruct references from objectStrings
                if (obj.__dict__[key].dtype.char == "S" or
                    obj.__dict__[key].dtype.char == "U" or
                    obj.__dict__[key].dtype.char == "O"):
                    references = np.empty_like(obj.__dict__[key], object)
                    iterator   = np.nditer(obj.__dict__[key],
                                           flags=['refs_ok','multi_index'])
                    while not iterator.finished:
                        idno = _idFromObjectString(iterator[0][()])
                        if idno is not None:
                            references[iterator.multi_index] = \
                                                objectlist[idlist.index(idno)]
                        iterator.iternext()
                    obj.__dict__[key] = references
            # Reconstruct references outside lists
            if (type(obj.__dict__[key]) == str or
               type(obj.__dict__[key]) == unicode):
                idno = _idFromObjectString(obj.__dict__[key])
                if idno is not None:
                    obj.__dict__[key] = objectlist[idlist.index(idno)]
                   
    # Find parent
    for obj in objectlist:
        try:
            if obj.parent is None:
                return obj
        except AttributeError:
            pass

def _idFromObjectString(string):
    """
    This method is used for validating a string describing an object and 
    returning the internal identification number of this object (this is 
    used in rebuilding the references in a JSON file).
    """
    if string is None:
        return None
    p = re.split("%%", string)
    if p is not None:
        if len(p) == 4 :
            if (p[0] == "PYVSIMOBJECT") and (p[2] == "IDNUMBER"):
                return int(p[3])
    return None

def _instantiateFromObjectString(string):
    """
    This method is capable of reading and validating a string describing 
    an object. If everything is ok, will return an instance of this object
    initialized without parameters.
    """
    p = re.split("%%", string)
    assert (p[0] == "PYVSIMOBJECT") 
    assert (p[2] == "IDNUMBER")
    p = re.split("[\.\']",p[1])
    assert (p[0] == "<class ")
    assert (p[4] == ">")
    pkg = __import__(p[1])
    mod = getattr(pkg,p[2])
    obj = getattr(mod,p[3])()
    return obj

def _objectString(obj):
    """
    Takes an object derived from the Core.PyvsimObject class and generates
    a string to identify it.
    """
    if obj is not None:
        return "PYVSIMOBJECT%%" + str(type(obj)) + "%%IDNUMBER%%" + str(obj.id)
    else:
        return None