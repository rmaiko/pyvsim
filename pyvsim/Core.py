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
import Utils
import couchdb
import System
import json

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
        
    def __repr__(self):
        """
        Generates a unique string to identify the object
        """
        return ("PYVSIMOBJECT%%" + str(type(self)) +
                "%%IDNUMBER%%" + str(self.id))

    
class Databasable(object):
    DB_URL  = Utils.readConfig("Database","databaseAddress")
    DB_USER = Utils.readConfig("Database","databaseUsername")
    DB_PASS = Utils.readConfig("Database","databasePassword")
    _COUCH = None

    def __init__(self):
        self.dbParameters = None
        self.dbName       = None
        self.name         = None
        self._db          = None
        
    @property
    def db(self):
        if Databasable._COUCH is None or self._db is None:
            self._initializeDB()
        return self._db
                         
    def _initializeDB(self):
        if Databasable._COUCH is None:
            print "Connecting to ", Databasable.DB_URL
            try:
                Databasable._COUCH = couchdb.Server(Databasable.DB_URL)
                Databasable._COUCH.resource.credentials = (Databasable.DB_USER,
                                                           Databasable.DB_PASS,)
            except couchdb.http.ServerError, e:
                raise e
            
        while not Databasable._COUCH.__nonzero__():
            print "Waiting"
            pass
        
        if self._db is None:
            self._db = Databasable._COUCH[self.dbName]
        
    def fetchFromDB(self, name):
        self._fromdb(self.db[name])
        self.name = name
        
    def listDB(self):
        string = ""
        for r in self.db.view("_all_docs"):
            string = string + r.key + "\n"
        return string
    
    def contributeToDB(self, overwrite = False):
        try:
            self.db[self.name] = self._dbdict()
        except couchdb.http.ResourceConflict:
            if overwrite:
                print "Overwriting existing entry"
                doc = self.db[self.name]
                self.db.delete(doc)
                self.db[self.name] = self._dbdict()
            else:
                print "Could not write to DB, probably doc already exists"
                
    def _dbdict(self):
        dbdict = {}
        for key in self.dbParameters:
            dbdict[key] = self.__dict__[key]
        sanitized = json.dumps(dbdict, cls = System.pyvsimJSONEncoder)
        return json.loads(sanitized)
    
    def _fromdb(self, dbdict):
        for key in self.dbParameters:
            if isinstance(dbdict[key], dict):
                self.__dict__[key] = json.loads(json.dumps(dbdict[key]),
                                                cls = System.pyvsimJSONDecoder)
            else:
                self.__dict__[key] = dbdict[key]