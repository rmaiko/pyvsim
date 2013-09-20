"""
.. module :: Core
    :platform: Unix, Windows
    :synopsis: Base classes for making pyvsim work
    
This module exists only to store the two classes that are used everywhere in
the program to define standard behaviors such as:

* Visitor pattern (for traversing object tree)
* Serialization (to files and to databases)
* Identification of objects (required by serializer)
    
.. moduleauthor :: Ricardo Entz <maiko at thebigheads.net>

.. license::
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

class PyvsimObject(object):
    """
    This is the base class of the program. It is used to implement the following
    behaviors:
    
    * Serialization (via the __getstate__ method)
    * Visitor pattern 
    * Identification and naming of objects
    """
    instanceCounter          = 0

    def __init__(self):
        self._id                           = PyvsimObject.instanceCounter
        
        #: Name of the object, used for displaying and referencing the object
        self.name                          = str(self._id)
        self.name = str(id(self))
        PyvsimObject.instanceCounter      += 1
        #: Fields to be excluded in the serialization process, empty list by default
        self.transientFields               = []
        
    @property
    def id(self):               return self._id

    def __getstate__(self):
        """
        This function provides the infrastructure for transient fields (those
        which are not serialized) which works both for builtin python pickle
        and pyvsimJSON.
        
        Returns
        -------
        mydict : dict
            Persistence dictionary, with transient fields set to None
        """
        mydict = self.__dict__
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
            an object inheriting from :doc:`Visitor`
        """
        visitor.visit(self)
        
    def __repr__(self):
        """
        Generates a unique string to identify the object
        """
        return (self.name+" - PYVSIMOBJECT%%" + str(type(self)) +
                "%%IDNUMBER%%" + str(self.id))

    
class PyvsimDatabasable(PyvsimObject):
    """
    This class provides another serialization method for some objects  - in a
    database.
    """
#    DB_OBJ  = Utils.readConfig("Database","databaseType")
#    DB_URL  = Utils.readConfig("Database","databaseAddress")
#    DB_USER = Utils.readConfig("Database","databaseUsername")
#    DB_PASS = Utils.readConfig("Database","databasePassword")

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
        if self._db is None:
            self._initializeDB()
        return self._db
                         
    def _initializeDB(self):
        """
        Gets the corresponding database facade (according to the config file),
        then initializes.
        """
        pkg         = __import__("DBFacade")
        #mod         = getattr(pkg,"DBFacade")
        self._db    = getattr(pkg,
              PyvsimDatabasable.DB_OBJ)(dburl    = PyvsimDatabasable.DB_URL, 
                                        dbName   = self.dbName, 
                                        username = PyvsimDatabasable.DB_USER, 
                                        password = PyvsimDatabasable.DB_PASS)
        
        
        
    def fetchFromDB(self, name):
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
        dbdict = self.db.fromdb(name)
        for key in self.dbParameters:
            self.__dict__[key] = dbdict[key]
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
        return self.db.listdb()
    
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
        ValueError
            When an entry with the same name already exists in the database
        """
        dbdict = self._dbdict()
        self.db.todb(self.name, dbdict, overwrite)
       
                
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
        return dbdict