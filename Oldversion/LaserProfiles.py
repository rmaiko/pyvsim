from __future__ import division
import numpy as np
import copy
import vtk
import pprint
import time
import threading
import vec

def prototypeIntensityFunction(x,y):
    """
    Placeholder function for laser intensity.
    
    When writing a function, please keep in mind that:
    - it must be defined in the domain [-1..1] for both variables
    - the center of the beam is in the point (0,0)
    
    The simulation expects that:
    [intensityFunction * intensityMultiplier] = Joules/m^2
    """
    raise NotImplementedError

def gaussianIntensity(x,y):
    """
    Gaussian laser intensity, just a testing function
    """
    return np.exp(-((x*5)**2+(y*5)**2))
 