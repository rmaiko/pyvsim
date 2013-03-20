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
import Primitives
import System
import Utils
import numpy as np
if __name__ == '__main__':
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
    import Curves
    sellmeierCoeffs      = np.array([[1.03961212, 6.00069867e-15],
                                     [0.23179234, 2.00179144e-14],
                                     [70.01046945, 1.03560653e-10]])
    vol.refractiveIndexLaw = Curves.SellmeierEquation(sellmeierCoeffs)
    
    r = Primitives.RayBundle()
    n = 100
    v = Utils.normalize(np.array([0.5,0.17,0]))
    p = np.array([-0.5,0.1,0.05])
    v = np.tile(v,(n,1))
    w = np.linspace(380e-9, 780e-9, n)
    r.insert(v, p, w)
    a = Primitives.Assembly()
    a.insert(vol)
    a.insert(r)
    r.maximumRayTrace = 2
    r.trace()
    System.plot(a,displayAxes=False)