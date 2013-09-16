'''
Created on Sep 4, 2013

@author: entz_ri
'''
import pyvsim
import json
import re
import argparse
import sys

def readscenario(filename):
    """
    Reads a JSON file containing a scenario structure, instantiates every
    object and saves it to another JSON file for human-readability.
    """
    f = open(filename)
    content = f.read()
    f.close()
    
    structure = json.loads(content)
    
    name      = structure["name"]
    del structure["name"]
    structure["type"] = "Assembly"
    
    env = popelements(name, structure)
    
#     pyvsim.System.save(env, name, "json")
    pyvsim.System.save(env, name)
    return env
    
def readinstructions(filename):
    """
    Reads a JSON file with instruction for loading a simulation and executing
    it. 
    """
    f = open(filename)
    content = f.read()
    f.close()
    
    structure = json.loads(content)
    
    environment = pyvsim.System.load(structure["scenario"])
    
    commands    = structure["commands"]
    
    for target, command in commands:
        action  = Command(name = target, commands = command)
        environment.acceptVisitor(action)
        if not action.executed:
            raise ValueError("Object " + action.name + " not found!")
    
    return environment
    
class Command(object):
    """
    This class exploits the visitor pattern to traverse the environment tree, 
    find an object by name and send it an order to execute a given method, or 
    a change of a given parameter.
    """
    def __init__(self, name, commands):
        self.executed = False
        self.name      = str(name)
        self.commands  = commands
        
    def visit(self, obj):
        if self.name == obj.name:
            configelement(obj, self.commands)
            self.executed = True
            
def popelements(key, dictionary):
    """
    Instantiates the objects in a scenario. It is restricted to objects from
    the :doc:`Toolbox` module, STL files and assemblies.
    
    Parameters
    ----------
    key : str
        The name of the object to be instantiated
    dictionary : dict
        A dictionary with the following mandatory keys:
            
            * type : "STL", "Assembly" or the name of the class in the Toolbox
              module
            * filename, **iif** type = "STL", is the name of the STL file to be
             read
    """
    print "Parsing ", key
    
    if dictionary["type"] == "Assembly":
        blob      = pyvsim.Primitives.Assembly()
        for name, value in dictionary.items():
            if name != "type":
                print "Recursive call to ", name
                blob += popelements(name, value)
                
    elif dictionary["type"] == "STL":
        blob      = pyvsim.Utils.readSTL(dictionary["filename"])
        print "Read ", dictionary["filename"]," ", blob, "\n\n"
        del dictionary["filename"]
        configelement(blob, dictionary)
        
    else:
        mod         = pyvsim.Toolbox
        blob        = getattr(mod, dictionary["type"])()
        configelement(blob, dictionary)
     
    blob.name   = str(key)
    return blob

def configelement(blob, configs):
    """
    This is the implementation that is capable of setting an attribute or 
    calling a function of an object or module
    
    Parameters
    ----------
    blob : object or module
        The place where the functions or attributes are (can be a module, or
        an object)
    configs : dict
        A dictionary which keys are the names of the attributes/functions and
        the values are either the value to be set (in case of attribute) or:
        
        * A value - in case the function takes only one parameter
        * A dict - in case several parameters are needed. In this case, only 
        named parameters are supported
        
    Raises
    ------
    AttributeError
        In case any name is not found
    """
    for key, item in configs.items():
        if key != "type":
            # Block to select correct property/function and produce 
            # meaningful results in case the blob is not found
            try:
                # Trick to dig into complex paths (e.g. Camera.lens.aperture)
                path = re.split("\.", key)
                configurable_n = blob
                for stuck in path:
                    configurable_n_1    = configurable_n
                    configurable_n      = getattr(configurable_n_1, stuck)
                # Configurable_n = the field we are looking for
                # Configurable_n_1 = parent of the field
                # We can't set a field directly, so we must remember who is the
                # parent
            except AttributeError as e:
                print "The attribute ", key, " was not found in ", blob
                raise e
            
            # Execute function or set attribute
            if hasattr(configurable_n, "__call__"):
                if type(item) is dict:
                    configurable_n(**item)
                else:
                    configurable_n(item)
            else:
                print "Setting ", key, " at ", configurable_n_1, " to ", item
                setattr(configurable_n_1, path[-1], item)
                
if __name__ == "__main__":
    args = sys.argv[1:]
    parser = argparse.ArgumentParser("Executes pyvsim in a command line interface")
    parser.add_argument("--geometry",
                        dest    = "run",
                        action  = "store_const",
                        const   = readscenario,
                        default = None,
                        help    = "Reads a scenario file and generates a pyvsim scenario file")
    parser.add_argument("--simulation",
                        dest    = "run",
                        action  = "store_const",
                        const   = readinstructions,
                        default = None,
                        help    = "Reads a pyvsim command file and execute the simulation")
    parser.add_argument("--filename",
                        type    = type("a"),
                        help    = "Filename",
                        required= True)
    parser.add_argument("--plot",
                        choices = [0,1],
                        type    = type(True),
                        help    = "Toggle plotting of the scenario")
    arguments = parser.parse_args(args)

    env = arguments.run(arguments.filename)
    if arguments.plot:
        pyvsim.System.plot(env)