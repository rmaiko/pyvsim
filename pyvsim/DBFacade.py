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
import couchdb
import json
import System

class Couchpyvsim(object):
    COUCH   = None
    TIMEOUT = 10
    
    def __init__(self, dburl, dbName, username, password):
        self.dburl      = dburl
        self.dbName     = dbName
        self.username   = username
        self.password   = password 
        self.db         = None
        self._initializeDB()
        
    def _initializeDB(self):
        if Couchpyvsim.COUCH is None:
            print "Connecting to DB ", self.dburl
            try:
                couch = couchdb.Server(self.dburl)
                couch.resource.credentials = (self.username, self.password)
            except couchdb.http.ServerError, e:
                raise e
            print "Connection open, waiting for answer",
            while not couch.__nonzero__():
                print ".",
            print "Connection successful"
            
            Couchpyvsim.COUCH = couch
        
        self.db  = Couchpyvsim.COUCH[self.dbName]
        
    def todb(self, name, dbdict, overwrite = False):
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
        ValueError
            When an entry with the same name already exists in the database
        """
        dbdict = json.dumps(dbdict, cls = System.pyvsimJSONEncoder)
        dbdict = json.loads(dbdict)
        
        try:
            self.db[name] = dbdict()
        except couchdb.http.ResourceConflict:
            if overwrite:
                print "Overwriting existing entry"
                doc = self.db[name]
                self.db.delete(doc)
                self.db[name] = dbdict()
            else:
                print "Could not write to DB, probably doc already exists"
                raise ValueError
            
    def fromdb(self, name):
        dbdict = self.db[name]
        for key in dbdict.keys():
            dbdict[key] = json.loads(json.dumps(dbdict[key]), 
                                     cls = System.pyvsimJSONDecoder)
        return dbdict
            
    def listdb(self):
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