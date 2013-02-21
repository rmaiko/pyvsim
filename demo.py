#!/usr/bin/env python
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

if __name__=="__main__":
    """
    This is a demo run of PyVSim
    """
    import Core
    import System
    import Utils
    import numpy as np
    
    # Creation of some simulation elements. 
    part        = Core.Volume() # Hexahedrons, usually initialized as cubes
    part2       = Core.Volume()
    bundle      = Core.RayBundle() # A container of rays
    assembly    = Core.Assembly()  # A container of parts
    
    # All parts must be contained in a "main" assembly. Assemblies can contain
    # sub-assemblies, no problem though
    assembly.insert(part)
    assembly.insert(bundle)
    assembly.insert(part2)
    
    # Let's create points to degenerate the hexahedron into a prism-like
    points = [[0,0,0],
              [0.001,0,0],
              [1,1,0],
              [0,1,0],
              [0,0,1],
              [0.001,0,1],
              [1,1,1],
              [0,1,1]]
    part.points = np.array(points)
    part.clearData()
    
    # Let's create some rays to do raytracing
    nrays = 100
    # The origin of the bundle is the natural place for rays to start,
    # if you specify otherwise, no problem
    bundle.translate(np.array([0.3,1.2,0.5]) - bundle.origin)
    # This defines the maximum raytracing distance we want
    bundle.maximumRayTrace = 5
    bundle.stepRayTrace    = 2
    # This creates a vector pointing to the prism
    vec = [0.55-0.3,1-1.2,0]
    vec = Utils.normalize(np.array(vec))
    vec = np.tile(vec, (nrays,1))
    # And here a list of wavelengths spanning a little more than the 
    # visible range
    wavelength = np.linspace(360e-9,800e-9,nrays)
    
    # Let's ask for the bundle to initialize the rays
    # Note the "None", it means that the rays start from the bundle origin
    bundle.insert(vec, None, wavelength)
    
    # These coefficients are used in the sellmeier equation
    # (this makes the index of refraction vary with wavelength)
    # 
    # The coefficients here are for BK7 crown glass, try playing around
    part.sellmeierCoeffs      = np.array([[1.03961212, 6.00069867e-15],
                                          [0.23179234, 2.00179144e-14],
                                          [1.01046945, 1.03560653e-10]])
    
    # Let's position the other cube at a more interesting position...
    part2.translate(np.array([1,0,0.5]))
    part2.indexOfRefraction = 1.666
    part2.rotate(np.pi/6, part2.x)
    
    # Not let's trace
    print "Tracing ray bundle"
    print "Pre allocated steps : ", bundle.preAllocatedSteps
    print "Step ray trace      : ", bundle.stepRayTrace
    tic = Utils.Tictoc()
    tic.tic()
    # The bundle knows where to trace because it is in the same
    # assembly as all the other components, do you get now why 
    # every part has to be inserted at the same assembly?
    bundle.trace()
    tic.toc()
    print "Number of steps     : ", bundle.steps
       
    # Now we ask for a plotter object
    # if you want to have some fun, try: System.Plotter("mpl")
    plotter = System.Plotter()
    # Now we make the plotter visit our assembly and gather the data
    # for plotting
    assembly.acceptVisitor(plotter)
    # Finally, display everything
    plotter.display()