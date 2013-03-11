'''
Created on 12 mars 2013

@author: maiko
'''
import Core
import System
import Utils
import numpy as np
if __name__ == '__main__':
    vol = Core.Volume()
    vol.points = np.array([[0   ,0,0],
                           [1   ,0,0],
                           [0.5 ,0.866,0],
                           [1e-6,0,0],
                           [0   ,0,0.1],
                           [1   ,0,0.1],
                           [0.5 ,0.866,0.1],
                           [1e-6,0,0.1]])
    vol.surfaceProperty = vol.TRANSPARENT
    vol.sellmeierCoeffs = np.array([[1.03961212, 6.00069867e-15],
                                    [0.23179234, 2.00179144e-14],
                                    [30.01046945, 1.03560653e-10]])
    r = Core.RayBundle()
    n = 100
    v = Utils.normalize(np.array([0.5,0.17,0]))
    p = np.array([-0.5,0.1,0.05])
    v = np.tile(v,(n,1))
    w = np.linspace(380e-9, 780e-9, n)
    r.insert(v, p, w)
    a = Core.Assembly()
    a.insert(vol)
    a.insert(r)
    r.maximumRayTrace = 2
    r.trace()
    System.plot(a,displayAxes=False)