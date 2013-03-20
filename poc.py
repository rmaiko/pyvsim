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
import couchdb

LALA = None

#couch = couchdb.Server("https://maiko.iriscouch.com/")
#print couch
#db    = couch["test_pyvsim"]
#for id in db:
#    print id
    
#import ConfigParser
#parser = ConfigParser.SafeConfigParser()
#parser.read("config.dat")
#print parser.sections()
#print parser.get("System", "databaseAddress")
#LALA = parser.get("System", "databaseAddress")
#print LALA
#
#Core.GLOBAL_TOL = parser.get("System", "databaseAddress")
#print Core.GLOBAL_TOL

if __name__ == '__main__':
    
    from math import *
    x = 532e-9 * 1e6
    print x
    print sqrt( 1 + 1.03961212*pow(x,2)/(pow(x,2)-0.00600069867) + 0.231792344*pow(x,2)/(pow(x,2)-0.0200179144) + 1.01046945*pow(x,2)/(pow(x,2)-103.560653) )
     
    import Curves
    eq = Curves.SellmeierEquation()
    print eq.eval(532e-9)
#db    = couch.create("test_pyvsim")
#print db
#db["foo"] = {1: 2}
#print db["foo"]


#env = Core.Assembly()
#c = Toolbox.Camera()
#v = Core.Volume()
#env.insert(c)
#env.insert(v)
#
#c.lens.translate(np.array([0.026474,0,0]))
#c.lens.focusingDistance = 1
#c.lens.aperture         = 2
#
#v.dimension = np.array([0.02, 1, 1])
#v.translate(np.array([0.35, 0, 0]))
#v.indexOfRefraction = 1.1 #5849
#v.surfaceProperty = v.TRANSPARENT
#v.rotate(np.pi*70/180, env.z)
#
#c.depthOfField()
#
#System.plot(env)