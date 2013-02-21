##!/usr/bin/env python
#"""
#PyVSim v.1
#Copyright 2013 Ricardo Entz
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#"""
#
#def erf(inp):
#    """
#    The algorithm of the error function comes from 
#    
#    Handbook of Mathematical Functions, formula 7.1.26.
#    http://www.math.sfu.ca/~cbm/aands/frameindex.htm
#    
#    Vectorized, in order to improve performance
#    """
#    # save the sign of x
#    # sign = np.sign(input)
#    x    = np.abs(inp)
#    
#    # constants
#    # a1 =  0.254829592
#    # a2 = -0.284496736
#    # a3 =  1.421413741
#    # a4 = -1.453152027
#    # a5 =  1.061405429
#    # p  =  0.3275911
#
#    # A&S formula 7.1.26
#    # t = 1.0/(1.0 + p*x)
#    t = 1.0 / (1.0 + 0.3275911*x)
#
#    # y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
#    y = 1.0 - (((((1.061405429*t -1.453152027)*t) \
#                   + 1.421413741)*t -0.284496736)*t \
#                   + 0.254829592)*t*np.exp(-x*x)
#
#    # y = y*sign
#    return y*np.sign(inp)
#
#def recordParticle(self,coords,energy,wavelength,diameter):
#    """
#    coords     - [y,z] parametric coordinates of the recorded point
#    energy     - J -   energy which is captured by the lenses
#    wavelength -       illumination wavelength
#    diameter   - m -   particle image diameter
#    
#    Writes the diffraction-limited image of a particle to the
#    sensor. This logic was created by Lecordier and 
#    Westerweel on the SIG
#    
#    Do not forget the sign convention for the image
#    --|----------------------------------> 
#      |                            |     z
#      |                            |
#      |            A____________   | 
#      |            |            |  |
#      |            |            |  |      A = anchor
#      |            |      C     |  |      C = particle center
#      |            |            |  | 
#      |            |____________|  |
#      v ___________________________| 
#       y
#       
#      When particle image is partially out of the image limits, the computation
#      is done over a partially useless domain, but remains correct
#    --|----------------------------------> 
#      |                            |     z
#      |                            |
#      |                            | 
#      |               A____________|
#      |               |            |      A = anchor
#      |               |            |      C = particle center
#      |               |            | 
#      |               |            |
#      v ______________|___________C| 
#       y
#    
#    The program performs an integration of a 2D gaussian distribution over the
#    sensitive areas of the pixel (defined by the fill ratio). 
#    
#    The static method erfnorm was defined because scipy was not available
#    
#    This function is NOT suited for calculating more than one particle at a 
#    time.
#    
#    """
#    if sum((coords < 0) + (coords > 1)):
#        return
#    # Classical formula  E =    h*c        h = 6.62607e-34 J*s
#    #                        ---------     c = 299 792 458 m/s
#    #                          lambda
#    photonEnergy = 6.62607e-34*299792458/wavelength
#    totalPhotons = energy / photonEnergy
#    # print "Total Photons ", totalPhotons
#    if totalPhotons < 2:
#        return
#    #
#    pixels   = self.parametricToPixel(coords) 
#    # this masksize guarantees that the particle image will not be cropped, when
#    # the particle size is too small
#    masksize = np.round(2.25*diameter/self.pixelSize) 
#    masksize = (masksize > 3)*masksize + 3*(masksize <= 3)
#    # masksize = np.round(10*diameter/self.pixelSize) 
#    # Defines the anchor position (cf. documentation above)
#    anchor   = np.round(pixels - masksize/2) 
#    # Important to verify if anchor will not force an incorrect matrix addition
#    anchor   = anchor * (anchor >= 0) * (anchor + masksize <= self.resolution) + \
#               (np.array([0,0])) * (anchor < 0) + \
#               (anchor-masksize) * (anchor + masksize > self.resolution)
#    [X,Y] = np.meshgrid(range(int(anchor[0]),int(masksize[0]+anchor[0])),
#                        range(int(anchor[1]),int(masksize[1]+anchor[1])))
#    # Magic Airy integral, in fact this is a 2D integral on the sensitive
#    # area of the pixel (defined by the fill ratio)
#    s = (diameter/self.pixelSize)*(0.44/1.22)
#    gx0 = 0.5*(erf((((X-pixels[0]) - 0.5*self.fillRatio[0])*2/s[0]) / \
#                                SQRT2) + 1)
#    gx1 = 0.5*(erf((((X-pixels[0]) + 0.5*self.fillRatio[0])*2/s[0]) / \
#                                SQRT2) + 1)
#    gy0 = 0.5*(erf((((Y-pixels[1]) - 0.5*self.fillRatio[1])*2/s[1]) / \
#                                SQRT2) + 1)
#    gy1 = 0.5*(erf((((Y-pixels[1]) + 0.5*self.fillRatio[1])*2/s[1]) / \
#                                SQRT2) + 1)
#
#    level = (gx1-gx0)*(gy1-gy0)*totalPhotons*self.quantumEfficiency
#    # print "Level matrix, rows: ", np.size(level,0), " columns: ", np.size(level,1)
#    # print level
#    self.rawData[anchor[0]:(anchor[0]+masksize[0]),anchor[1]:(anchor[1]+masksize[1])] = \
#        self.rawData[anchor[0]:(anchor[0]+masksize[0]),anchor[1]:(anchor[1]+masksize[1])] + \
#        level