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

This file lists the objects to be used in a normal execution of pyvsim. This
way, the user does not have to know the internal distribution of classes (which
may look somewhat arbitrary, e.g. Volume is in Primitives, not in Toolbox)

The down side is that one needs to register the classes here when the user
has to have access
"""
import Core
import Primitives
import Toolbox
import Utils
import System
import Library


class Pyvsim(object):
    def __init__(self):
        self.scenario = Primitives.Assembly()
        
    def __getitem__(self, k):
        """
        """        
        if type(k) is str:
            for item in self.scenario._items:
                if item.name == k:
                    return item
        else:
            return self.scenario._items[k]
    
    def __setitem__(self, k, value):     
        """
        """        
        self.scenario.append(value, k)      
        
    def __delitem__(self,k):      
        """
        """        
        self.scenario.remove(k)
        
    def __len__(self):
        """
        """        
        return len(self.scenario._items)
    
    def __contains__(self, other):
        """
        """
        if type(other) is str:
            for item in self.scenario._items:
                if item.name == other:
                    return True
            return False
        return other in self._items
                
    def importgeometry(self, filename):
        """
        Reads a STL file and adds it to the pyvsim scenario
        """
        part                    = Utils.readSTL(filename)
        self.scenario          += part
        self.objects[part.name] = part
        
    def save(self, filename, mode = "pickle"):
        """
        """        
        System.save(self.scenario, filename, mode)
        
    def load(self, filename, overwrite = False):
        """
        
        """
        if not overwrite and len(self.scenario) > 0:
            ans = raw_input("Current scenario is not empty, overwrite? (y/n)")
            if ans.upper() != "Y":
                print "Aborted readind file ", filename
                return
        self.scenario = System.load(filename)

"""
Initialization of pyvsim
"""
print "Loading pyvsim"
pyvsim = Pyvsim()
print "done!"