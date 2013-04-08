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

class Material(Core.PyvsimDatabasable):
    def __init__(self):
        Core.PyvsimObject.__init__(self)
        Core.PyvsimDatabasable.__init__(self)
        self.name           = 'Material property '+str(self._id)
        self.source         = "None"   
        
    def refractiveIndex(self, wavelength, position = None):
        raise NotImplementedError

class IdealMaterial(Material):
    def __init__(self, value = 1):
        Material.__init__(self)
        self.dbParameters   = ["name",
                               "value"]
        self.name                     = 'Ideal Material'
        self.source                   = "constant refractive index"
        self.value                    = value
        
    def refractiveIndex(self,wavelength):            
        return self.value

class Glass(Material):
    def __init__(self, coeffs = None):
        Material.__init__(self)
        self.dbName     = "glass"
        self.dbParameters   = ["name",
                               "source",
                               "refractiveIndexConstants"]
        self.name    = 'Schott N-BK7 Borosilicate Crown Glass'    
        if coeffs is None:
            # Add coefficients of BK7 crown glass
            self.source  = "http://www.us.schott.com/advanced_optics/english/download/index.html"
            coeffs = np.array([[1.03961212, 0.00600069867],
                               [0.23179234, 0.02001791440],
                               [1.01046945, 103.560653000]])
        self.refractiveIndexConstants = coeffs
        
    def refractiveIndex(self,wavelength):
        Nc = np.size(self.refractiveIndexConstants,0)
        w2 = (wavelength*1e6) ** 2                 
        return np.sqrt(1 +
               np.sum((w2 * self.refractiveIndexConstants[:,0].reshape(Nc,1,1)) /
                      (w2 - self.refractiveIndexConstants[:,1].reshape(Nc,1,1)),
                          0)).squeeze()

class Plastic(Material):
    def __init__(self, coeffs = None):
        Material.__init__(self)
        self.dbName         = "plastic"
        self.dbParameters   = ["name",
                               "source",
                               "refractiveIndexConstants"] 
        self.name           = "PMMA a.k.a. acrylic"
        if coeffs is None:
            # Add coefficients of BK7 crown glass
            self.source  = ("Sultanova, N.; Kasarova, S. & Nikolov, I. " +
                            "Dispersion properties of optical polymers " +
                            "Acta Physica Polonica, A, 2009, 116, 585-587")
            coeffs = np.array([ 2.3999964,
                               -8.308636e-2, 
                               -1.919562e-1,
                                8.720608e-2,
                               -1.666411e-2,
                                1.169519e-3])
        self.refractiveIndexConstants = coeffs
        
    def refractiveIndex(self,wavelength):   
        wavelength = wavelength*1e6 
        return np.sqrt(self.refractiveIndexConstants[0] +
                       self.refractiveIndexConstants[1]*(wavelength**2) +
                       self.refractiveIndexConstants[2]*(wavelength**-2) +
                       self.refractiveIndexConstants[3]*(wavelength**-4) +
                       self.refractiveIndexConstants[4]*(wavelength**-6) +
                       self.refractiveIndexConstants[5]*(wavelength**-8))


    