#!/usr/bin/python
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

"""
This is a demo of field of view calculation in an airplane mockup.
"""
import sys
sys.path.append("../")
import numpy as np
import pyvsim.Utils
import pyvsim.Toolbox
import pyvsim.Primitives
import pyvsim.System

if __name__ == "__main__":
    # Just importing something to time the procedures
    tic = pyvsim.Utils.Tictoc()
    
    # Reads the airplane model from the STL file
    tic.tic()
    part = pyvsim.Utils.readSTL("halfmodel.stl")
    tic.toc()
    
    # Start timing the field of view calculation
    tic.tic()
    # The part was set as transparent, because the camera rays must pass through
    # it in order to be able to create a mapping
    part.surfaceProperty = part.TRANSPARENT
    # Only visualization properties, they don't affect calculation
    part.opacity = 1
    part.color = np.array([1,1,1])
    
    # Create a camera
    c = pyvsim.Toolbox.Camera()
    ## This offset is because we're using a lens with Canon mount in a C-mount camera
    #c.lens.translate(np.array([0.026474,0,0]))
    # This is to rotate the lens with respect to the camera axis, so we can have
    # Scheimpflug condition
    c.setScheimpflugAngle(-0.05, c.z)
    c.lens.aperture = 5.6
    # These positions are taken from CATIA
    windowCenter = np.array([1.711, -3.275, 0.75])
    wingTarget   = np.array([7, -0.921, -1.404])
    # Create a vector with the direction the camera should be pointing
    v = wingTarget - windowCenter
    # Create a vector perpendicular to v
    vp = np.array([-v[1]*v[2], 
                   -v[0]*v[2], 
                   2*v[0]*v[1]])
    # Makes the camera align to the vectors we've created
    c.alignTo(v, vp)  
    # Moves the camera to the window
    c.translate(windowCenter - c.origin)
    # Creates an assembly. Everything must be inside a single assembly to run
    # properly
    a = pyvsim.Primitives.Assembly()
    # Insert the airplane and the camera
    a.append(part)
    a.append(c)
    # The mapping resolution must be more than [2, 2] if there is refraction in 
    # the way (because it could distort the field of view), in this case we put
    # [10, 10] just to check performance
    c.mappingResolution = [10, 10]
    # This is the focus setting as written on the lens
    c.lens.focusingDistance = 5.2
    # This is makes the camera calculate its depth of field and mapping
    c.initialize()
    # Stop timing
    tic.toc()
    # This is to display the scenario
    pyvsim.System.plot(a)