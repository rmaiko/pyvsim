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
        self.transientFields               = []
        
    @property
    def id(self):               return self._id

    def __getstate__(self):
        mydict = self.__dict__
        if self.transientFields is not None:
            for key in self.transientFields:
                mydict[key] = None
        return mydict
    
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

    
class PyvsimDatabasable(PyvsimObject):
    DB_URL  = Utils.readConfig("Database","databaseAddress")
    DB_USER = Utils.readConfig("Database","databaseUsername")
    DB_PASS = Utils.readConfig("Database","databasePassword")
    _COUCH  = None

    def __init__(self):
        self.dbParameters       = None
        self.dbName             = None
        self.name               = None
        self._db                = None
        self.transientFields.extend(["_db"])
        
    @property
    def db(self):
        """
        Returns the database object where libraries of parameters are stored.
        """
        if PyvsimDatabasable._COUCH is None or self._db is None:
            self._initializeDB()
        return self._db
                         
    def _initializeDB(self):
        if PyvsimDatabasable._COUCH is None:
            print "Connecting to ", PyvsimDatabasable.DB_URL
            try:
                PyvsimDatabasable._COUCH = couchdb.Server(PyvsimDatabasable.DB_URL)
                PyvsimDatabasable._COUCH.resource.credentials = \
                    (PyvsimDatabasable.DB_USER, PyvsimDatabasable.DB_PASS)
            except couchdb.http.ServerError, e:
                raise e
            
        while not PyvsimDatabasable._COUCH.__nonzero__():
            print "Waiting"
            pass
        
        if self._db is None:
            self._db = PyvsimDatabasable._COUCH[self.dbName]
        
    def fetchFromDB(self, name):
        """
        
        """
        self._fromdb(self.db[name])
        self.name = name
        
    def listDB(self):
        """
        Returns a list listing the current database entries in the category of 
        the object. E.g.: using this method in a Glass material will list only
        the available glasses, etc.
        
        Returns
        -------
        dblist : list
            List of strings containing all entries in a database category
        """
        dblist = []
        for r in self.db.view("_all_docs"):
            dblist.append(r.key)
        return dblist
    
    def contributeToDB(self, overwrite = False):
        """
        Contributes to the database with the current object parameters. By 
        default no overwriting is allowed. Each entry is defined by the 
        "name" field.
        
        Parameters
        ----------
        overwrite : boolean
            If False, will not allow an entry in the database to be modified.
            
        Raises
        ------
        couchdb.http.ResourceConflict
            When an entry with the same name already exists in the database
        """
        try:
            self.db[self.name] = self._dbdict()
        except couchdb.http.ResourceConflict, err:
            if overwrite:
                print "Overwriting existing entry"
                doc = self.db[self.name]
                self.db.delete(doc)
                self.db[self.name] = self._dbdict()
            else:
                print "Could not write to DB, probably doc already exists"
                raise err
                
    def _dbdict(self):
        """
        Created a dict only with the object entries that should be stored in 
        the database. These entries are defined in self.dbParameters.
        
        Returns
        -------
        dbdict : dict
            A JSON serializable dict created with System.pyvsimJSONEncoder with
            entries defined in self.dbParameters
        """
        dbdict = {}
        for key in self.dbParameters:
            dbdict[key] = self.__dict__[key]
        sanitized = json.dumps(dbdict, cls = System.pyvsimJSONEncoder)
        return json.loads(sanitized)
    
    def _fromdb(self, dbdict):
        """
        Redefines the current object with data received from the database in 
        the form of a dictionary.
        
        Parameters
        ----------
        dbdict : dict
            A dict received from the database and decodable by 
            pyvsimJSONDecoder. The dict must have all the fields defined in
            self.dbParameters
            
        Raises
        ------
        KeyError
            If the received dict doesn't have all the parameters defined in
            self.dbParameters
        """
        for key in self.dbParameters:
            if isinstance(dbdict[key], dict):
                self.__dict__[key] = json.loads(json.dumps(dbdict[key]),
                                                cls = System.pyvsimJSONDecoder)
            else:
                self.__dict__[key] = dbdict[key]