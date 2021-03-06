#!/usr/bin/python
"""
PyVSim part2.1
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

if __name__ == '__main__':
    import sys
    sys.path.append("../")
    import numpy as np
    from pyvsim import *
    """
    This demo shows a simple render of a famous image, but with 
    physically correct angles
    """
    vol = Primitives.Volume()
    vol.points = np.array([[0   ,0,0],
                           [1   ,0,0],
                           [0.5 ,0.866,0],
                           [1e-6,0,0],
                           [0   ,0,0.1],
                           [1   ,0,0.1],
                           [0.5 ,0.866,0.1],
                           [1e-6,0,0.1]])
    vol.surfaceProperty = vol.TRANSPARENT
    sellmeierCoeffs     = np.array([[1.03961212, 0.00600069867],
                                    [0.23179234, 0.02001791440],
                                    [70.01046945, 103.560653000]])
    vol.material = Library.Glass(sellmeierCoeffs)
    vol.material.name = "The dark side of the moon glass"
    
    r = Primitives.RayBundle()
    n = 200
    v = Utils.normalize(np.array([0.5,0.17,0]))
    p = np.array([-0.5,0.1,0.05])
    v = np.tile(v,(n,1))
    w = np.linspace(380e-9, 780e-9, n) #all the visible spectrum
    r.append(v, p, w)
    
    a = Primitives.Assembly()
    a.append(vol)
    a.append(r)
    
    r.maximumRayTrace = 2
    r.trace()
    
    System.save(obj = a, filename = "./test.dat", mode = "json")

    dec = System.load(filename = "./test.dat")
    
    System.plot(dec,displayAxes=False)