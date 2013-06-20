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
#        
#    def vars(self):
#        md = self.__dict__
#        md["lala"] = "lele"
#        return md
#   
#import pprint     
#print dir(test)
#mt = test()
#print vars(mt)
#        
#t1 = test()
#print t1.points
#
#t2 = test2()
#print t2.points(3)

import numpy as np
import Utils
import System
import Core
import Toolbox
import couchdb
import Library
import pprint
import json
import threading
import Primitives

#mat = Library.Glass()
#mat.fetchFromDB("Schott N-BK7")
#print mat.listDB()

import threading
import thread, time

#class dumm(object):
#    def __init__(self, name):
#        self.name = name
#
#class ov(object):
#    def __init__(self):
#        self._items = []
#        
#    def __iadd__(self,other):
#        if not issubclass(type(other), dumm):
#            raise TypeError("Operations are only allowed between \
#                             pyvsim components")       
#        self._items.append(other)       
#        return self
#    
#    def __isub__(self,other):
#        if not issubclass(type(other), dumm):
#            raise TypeError("Operations are only allowed between \
#                             pyvsim components")
#        self.remove(other)        
#        return self       
#        
#    def __eq__(self, other):
#        answer = np.zeros_like(other)
#        
#        answer += (other is self)
#
#        for item in self._items:
#            answer += (item == other)
#            
#        return answer
#    
#    def __neq__(self, other):
#        return 1 - (self == other) 
#        
#    def __getitem__(self, k):
#        print "gotitem"
#        if type(k) is str:
#            for item in self._items:
#                if item.name == k:
#                    return item
#        else:
#            return self._items[k]
#    
#    def __setitem__(self, k, value):     
#        print "setitem"
#        self.append(value, k)      
#        
#    def __delitem__(self,k):      
#        self.remove(k)
#        
#    def __len__(self):
#        return len(self._items)
#    
#    def __contains__(self, other):  
#        return other in self._items
#        
#    
#d1 = dumm("a")
#d2 = dumm("b")
#poc  = ov()
#poc += d1
#poc += d2

#print "Finished loading POC module"
#
#class timeoutWrapper():
#    def __init__(self, function, timeout = 10, forgiving = True):
#        self.function   = function
#        self.maxtime    = timeout
#        self.forgiving  = forgiving
#            
#    def timeout(self):
#        thread.interrupt_main()
#        
#    def run(self, *args, **kwargs):
#        try:
#            timer = threading.Timer(self.maxtime, self.timeout)
#            timer.start()
#            answer = self.function(*args, **kwargs)
#            timer.cancel()
#            return answer
#        except KeyboardInterrupt:
#            if self.forgiving:
#                print "Function timed out ", self.function
#            else:
#                raise
#            
#
#def myfun(t):
#    print 'it keeps going and going',
#    k = 1
#    while k < 500:
#        print '.',
#        time.sleep(t)
#        k = k + 1     
#    return k
#
#print Utils.timeout(myfun, timeout = 1, forgiving = False).run(0.01)
#        
#print timeoutWrapper(myfun, 3).run(0.1)
#  
#print "lalala"            
        
    

#print plastic._dbdict()
#print plastic.listDB()
#plastic.fetchFromDB("PMMA a.k.a. acrylic")
#
#print plastic.refractiveIndex(532e-9)
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

#if __name__ == '__main__':
#    couch = couchdb.Server("http://maiko.iriscouch.com")
#    db    = couch["test_pyvsim"]
#    print db
#    db["foo"] = {1: 2}
#    print db["foo"]

#env = Core.Assembly()
#c = Toolbox.Camera()
#v = Core.Volume()
#env.append(c)
#env.append(v)
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

import numpy as np
import Primitives
import Utils
import System

nsteps      = 2e3
dt          = 1e-3
velocity    = np.zeros((nsteps,3))
position    = np.zeros((nsteps,3))
accel       = np.array([0,-10,0])
velocity[0] = np.random.rand(3)
position[0] = np.random.rand(3)

for t in np.arange(nsteps-1):
    velocity[t+1] = velocity[t] + accel*dt
    if velocity[t+1,1] < 0 and position[t,1] < 0:
        velocity[t+1,1] = -0.8*velocity[t+1,1]
    position[t+1] = position[t] + velocity[t]*dt     
    
line            = Primitives.Line()
line.points     = position
vmag            = np.sqrt(np.sum(velocity*velocity,1))
line.color      = Utils.jet(vmag, np.min(vmag), np.max(vmag))
line.width      = 2
System.plot(line)