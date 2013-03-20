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
import numpy as np
import Core

class Constant(Core.PyvsimObject):
    def __init__(self, value = 1):
        Core.PyvsimObject.__init__(self)
        self.name                     = 'Constant '+str(self._id)
        self.value                    = value
        
    def eval(self,wavelength):            
        return self.value

class SellmeierEquation(Core.PyvsimObject):
    def __init__(self, coeffs = None):
        Core.PyvsimObject.__init__(self)
        self.name                     = 'Sellmeier Equation '+str(self._id)
        if coeffs is None:
            coeffs = np.array([[1.03961212, 6.00069867e-15],
                               [0.23179234, 2.00179144e-14],
                               [1.01046945, 1.03560653e-10]])
        self.refractiveIndexConstants = coeffs
        
    def eval(self,wavelength):
        Nc = np.size(self.refractiveIndexConstants,0)
        w2 = wavelength ** 2                 
        return np.sqrt(1 +
               np.sum((w2 * self.refractiveIndexConstants[:,0].reshape(Nc,1,1)) /
                      (w2 - self.refractiveIndexConstants[:,1].reshape(Nc,1,1)),
                          0)).squeeze()

class KasarovaEquation(Core.PyvsimObject):
    def __init__(self):
        Core.PyvsimObject.__init__(self, coeffs = None)
        self.name                     = 'Kasarova Equation '+str(self._id)
        self.refractiveIndexConstants = coeffs
        
    def eval(self,wavelength):             
        return np.sqrt(self.refractiveIndexConstants[0] +
                       self.refractiveIndexConstants[1]*wavelength**2 +
                       self.refractiveIndexConstants[1]*wavelength**-2 +
                       self.refractiveIndexConstants[1]*wavelength**-4 +
                       self.refractiveIndexConstants[1]*wavelength**-6 +
                       self.refractiveIndexConstants[1]*wavelength**-8)


    