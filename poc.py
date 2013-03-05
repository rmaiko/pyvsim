#!/usr/bin/env python
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
Just a sandbox for trying code snippets that I'm not sure how they 
work.
"""
class test(object):
    def __init__(self):
        self.points = [1,2,3]
#        
class test2(test):
    def __init__(self):
        test.__init__(self)
        self._other = "other"
        self._points = 1
        
    @property
    def points(self, test = 1):
        print "Activated getter"
        return self._points + test
    
    @points.setter
    def points(self,v):
        print "Activated setter"
        self._points = v
        
    @property
    def other(self):
        print "Other getter"
        return self._other
    
    @other.setter
    def other(self,v):
        print "Other setter"
        self._other = v
#        
#t1 = test()
#print t1.points
#
t2 = test2()
print t2.points(3)
#t2.points = [2,3,4]
#print t2.points

#import numpy as np
##M = np.random.rand(3,3,3)
##print M
##it = np.nditer(M, flags=['multi_index'])
##while not it.finished:
##    print it[0], it.multi_index
##    M[it.multi_index] = 1
##    it.iternext()
##print M
##
#ints =[[[None],
#        [None],
#        [None],
#        [None]],
#       [[1],
#        [1],
#        [1],
#        [1]],
#       [[1],
#        [0],
#        [1],
#        [0]],
#       [[0],
#        [1],
#        [1],
#        [1]]]
#
#coords = [[[  0.5    ,      0.5    ,      0.5       ],
#           [  0.5    ,      0.5    ,      0.5       ],
#           [  0.5    ,      0.5    ,      0.5       ],
#           [  0.5    ,      0.5    ,      0.5       ]],
#          [[  1.         ,  1.         ,  0.5       ],
#           [  0.         ,  1.         ,  0.5       ],
#           [  0.         ,  1.         ,  0.5       ],
#           [  0.5        ,  0.5        ,  1.        ]],
#          [[  4.53553391 ,  4.53553391 ,  0.5       ],
#           [ -3.53553391 ,  4.53553391 ,  0.5       ],
#           [ -3.53553391 ,  4.53553391 ,  0.5       ],
#           [  0.5        ,  0.5        ,  6.        ]],
#          [[  7.57106781 ,  7.57106781 ,  0.5       ],
#           [ -6.57106781 ,  7.57106781 ,  0.5       ],
#           [ -6.57106781 ,  7.57106781 ,  0.5       ],
#           [  0.5        ,  0.5        , 10.5       ]]]
#
#coords = np.array(coords)
#       
#ints = np.tile(ints,3) == 1
#firstInts = np.zeros_like(coords[0])
#firstMask = np.ones_like(coords[0])
#lastInts  = np.zeros_like(coords[0])
#
#for n in range(np.size(coords,0)):
#    #firstInts = firstInts + (firstMask * ints[n]) * coords[n]
#    firstInts[firstMask * ints[n] == 1] = coords[n][firstMask * ints[n] == 1]
#    firstMask = (ints[n] == 0) * (firstMask == 1)
#    lastInts[ints[n] == 1] = coords[n,ints[n] == 1]
#    
#print np.reshape(firstInts,(2,2,3)) 
#print firstInts
#
#    
##ph  = np.array([0,0,-300])
##pt1 = np.array([[0,0,0],
##                [0,1,0],
##                [1,1,0],
##                [1,0,0]])
##pt2 = pt1 + 0.00333333333*(pt1 - ph)
##xyz = np.vstack([pt1, pt2]) - np.array([0,0,0.5])
##uv = np.array([[0,0],
##               [0,1],
##               [1,1],
##               [1,0]])
##uv = np.vstack([uv, uv])
##print xyz
##print uv
##import Utils
##m = calculateDLT(uv, xyz)
##print m