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

part = Utils.readSTL("halfmodel.stl")
part.surfaceProperty = part.TRANSPARENT
part.opacity = 1
part.color = np.array([1,1,1])

c = Toolbox.Camera()
c.lens.translate(np.array([0.026474,0,0]))
c.lens.rotate(-0.05, c.z)
v = np.array([7, -0.921, -1.404]) - np.array([1.711, -3.275, 0.75])
vp = np.array([-v[1]*v[2], 
               -v[0]*v[2], 
               2*v[0]*v[1]])

vx = Utils.normalize(v)
vy = Utils.normalize(vp)
c.alignTo(vx, vy)  
c.translate(np.array([1.711, -3.275, 0.75]))

a = Core.Assembly()
a.insert(part)
a.insert(c)
c.mappingResolution = [10, 10]
c.lens.focusingDistance = 5.2

c.calculateMapping(part)

c.depthOfField()

System.plot(a)