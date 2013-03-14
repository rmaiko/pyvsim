#!/usr/bin/python
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
#class test(object):
#    def __init__(self):
#        self.points = [1,2,3]
##        
#class test2(test):
#    def __init__(self):
#        test.__init__(self)
#        self._other = "other"
#        self._points = 1
#        
#    @property
#    def points(self, test = 1):
#        print "Activated getter"
#        return self._points + test
#    
#    @points.setter
#    def points(self,v):
#        print "Activated setter"
#        self._points = v
#        
#    @property
#    def other(self):
#        print "Other getter"
#        return self._other
#    
#    @other.setter
#    def other(self,v):
#        print "Other setter"
#        self._other = v
##        
##t1 = test()
##print t1.points
##
#t2 = test2()
#print t2.points(3)

import numpy as np
import Utils
import System
import Core
import Toolbox

v = np.array([1,0,0])
v = np.tile(v,(2,1))
print Utils.rotateVector(v, 0.0, np.array([0,1,0]))