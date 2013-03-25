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
import copy
import numpy as np

class PyvsimObject(object):
    instanceCounter          = 0

    def __init__(self):
        self._id                           = PyvsimObject.instanceCounter
        self.name                          = str(self._id)
        PyvsimObject.instanceCounter      += 1
        
    @property
    def id(self):               return self._id
    
    def acceptVisitor(self, visitor):
        """
        This method is a provision for the `Visitor Pattern 
        <http://http://en.wikipedia.org/wiki/Visitor_pattern>`_ and is used
        for traversing the tree.
        
        Some possible uses are the display or the saving routine.
        
        *If you are inheriting from this class* and your node is non-terminal,
        please override this method
        
        Parameters
        ----------
        visitor 
            an object inheriting from `:class:~System.Visitor`
        """
        visitor.visit(self)
    
#    def sanedict(self):
#        sanedict = copy.deepcopy(self.__dict__) 
#        for k in sanedict.keys():
#            saneobject = sanedict[k]
#            if isinstance(sanedict[k], PyvsimObject):
#                saneobject = sanedict[k].__repr__()
#                
#            if isinstance(sanedict[k], np.ndarray):
#                if sanedict[k].dtype == np.dtype(object):    
#                    for element in np.nditer(sanedict[k], 
#                                             flags=['refs_ok'],
#                                             op_flags=['readwrite']):
#                        element[...] = element[()].__repr__()          
#                saneobject = sanedict[k].tolist()    
#                
#            sanedict[k] = saneobject
#             
#        return sanedict
        
    def __repr__(self):
        """
        Takes an object derived from the Core.PyvsimObject class and generates
        a string to identify it.
        """
        return ("PYVSIMOBJECT%%" + str(type(self)) +
                "%%IDNUMBER%%" + str(self.id))

    
class Databasable(object):
    def __init__(self):
        self.dbParameters = None
    
    def dbdict(self):
        dbdict = {}
        for key in self.dbParameters:
            dbdict[key] = self.__dict__[key]
        return dbdict
    
    def fromdb(self, dbdict):
        for key in dbdict.keys():
            self.__dict__[key] = dbdict[key]