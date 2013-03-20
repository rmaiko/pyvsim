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

def constant(parameter):
    def constant(x):
        return parameter
    return constant

def gaussian1d(parameter):
    def gaussian1d(x):
        return parameter[0]*np.exp(-(x-parameter[1])**2/(parameter[2]))
    return gaussian1d

def sellmeierEquation(sellmeierCoeffs):
    Nc = np.size(sellmeierCoeffs,0)
    def sellmeierEquation(wavelength):
        w2 = wavelength ** 2                 
        return np.sqrt(1 +
                   np.sum((w2 * sellmeierCoeffs[:,0].reshape(Nc,1,1)) /
                          (w2 - sellmeierCoeffs[:,1].reshape(Nc,1,1)),
                          0)).squeeze()
    return sellmeierEquation


    