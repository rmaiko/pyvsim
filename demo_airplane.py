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
import numpy as np
import Utils
import System
import Core
import Toolbox
# Just importing something to time the procedures
tic = Utils.Tictoc()

# Reads the airplane model from the STL file
tic.tic()
part = Utils.readSTL("halfmodel.stl")
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
c = Toolbox.Camera()
# This offset is because we're using a lens with Canon mount in a C-mount camera
c.lens.translate(np.array([0.026474,0,0]))
# This is to rotate the lens with respect to the camera axis, so we can have
# Scheimpflug condition
c.lens.rotate(-0.05, c.z)
# These positions are taken from CATIA
windowCenter = np.array([1.711, -3.275, 0.75])
wingTarget   = np.array([7, -0.921, -1.404])
# Create a vector with the direction the camera should be pointing
v = wingTarget - windowCenter
# Create a vector perpendicular to v
vp = np.array([-v[1]*v[2], 
               -v[0]*v[2], 
               2*v[0]*v[1]])
# We must normalize the vectors to use them
vx = Utils.normalize(v)
vy = Utils.normalize(vp)
# Makes the camera align to the vectors we've created
l = Toolbox.Laser()
l.translate(windowCenter - l.origin - np.array([0.1,-2.5,0]))
c.alignTo(vx, vy)  
# Moves the camera to the window
c.translate(windowCenter - c.origin)
# Creates an assembly. Everything must be inside a single assembly to run
# properly
a = Core.Assembly()
# Insert the airplane and the camera
a.insert(part)
a.insert(c)
a.insert(l)
l.trace()
# The mapping resolution must be more than [2, 2] if there is refraction in 
# the way (because it could distort the field of view), in this case we put
# [10, 10] just to check performance
c.mappingResolution = [10, 10]
# This is the focus setting as written on the lens
c.lens.focusingDistance = 5.2
# This is used to create a mapping for the camera, not useful here, but just
# demonstrated
c.calculateMapping(part)
# This makes the camera calculate (and display) its depth of field
c.depthOfField()
# Stop timing
tic.toc()

# This is to display the scenario
System.plot(a)