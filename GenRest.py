"""
.. module :: GenRest
    :platform: Unix, Windows
    :synopsis: Generates restructured text files from the modules in pyvsim
    
This module was implemented because it is very hard to generate a few dozen
rst files each time the documentation has to be compiled. This won't be 
documented now.
    
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
import os
import sys
import re


def xupadir(inpath, outpath, forbidden = [], package = None):
    sys.path.insert(0, os.path.abspath(inpath))
    files = os.listdir(inpath)
    for filename in files:
        if re.match(".*\.py$", filename) is not None:
            print "Generating ", filename
            chupatudo(inpath + "/" + filename, outpath, package)
                    
        if os.path.isdir(inpath + "/" + filename):
            if not (filename in forbidden):
                if package is not None:
                    xupadir(inpath + "/" + filename, 
                            outpath, forbidden, package+"."+filename)
                else:
                    xupadir(inpath + "/" + filename, 
                            outpath, forbidden, filename)
                        
class module_file(object):
    def __init__(self, package, name, outpath):
        self.name = name
        fname = name + ".rst"
        self.file = open(os.path.abspath(outpath+"/"+fname),"w")
        # Create the header
        self.file.write("="*len(name)+"\n")
        self.file.write(name+"\n")
        self.file.write("="*len(name)+"\n\n")
        if package is not None:
            self.file.write(".. currentmodule:: "+package+"."+name+"\n\n")
            self.file.write(".. automodule:: "+package+"."+name+"\n\n")
        else:
            self.file.write(".. currentmodule:: "+name+"\n\n")
            self.file.write(".. automodule:: "+name+"\n\n")
        self.file.write("Contents\n")
        self.file.write("========\n")
#        self.file.write(".. autosummary::\n")
        self.file.write(".. toctree::  \n")
        self.file.write("    :maxdepth: 2  \n\n")
        
    def addthing(self, name): 
        self.file.write("    " + name + "\n")
        
    def flush(self):
        self.file.close()
        
class class_file(object):
    def __init__(self, package, name, outpath):
        self.name = name
        fname = name + ".rst"
        self.file = open(os.path.abspath(outpath+"/"+fname),"w")
        # Create the header
        self.file.write("="*len(name)+"\n")
        self.file.write(name+"\n")
        self.file.write("="*len(name)+"\n\n")
        self.file.write(".. currentmodule:: "+package+"\n\n")
        self.file.write(".. autoclass:: "+package+"."+name+"\n")
        self.file.write("    :members: \n")            
                
    def flush(self):
        self.file.close()
        
class func_file(object):
    def __init__(self, package, name, outpath):
        self.name = name
        fname = name + ".rst"
        self.file = open(os.path.abspath(outpath+"/"+fname),"w")
        # Create the header
        self.file.write("="*len(name)+"\n")
        self.file.write(name+"\n")
        self.file.write("="*len(name)+"\n\n")
        self.file.write(".. currentmodule:: "+package+"\n\n")
        self.file.write(".. autofunction:: "+package+"."+name+"\n\n")  
                
    def flush(self):
        self.file.close()        
                
def chupatudo(filename, outpath, package = None):
    module_name = re.search("(.*)(\.py)", 
                            os.path.split(filename)[1]).group(1)
    print module_name
    
    f = open(filename, "r")
    indent = None
    
    module = module_file(package, module_name, outpath)
    
    for line in f:
        if re.match("\s*class .*",line) is not None:
            level = len(re.split("c+", line)[0])
            if level > 0:
                if indent is None:
                    indent = level
                level = level / indent
            name  = re.search("(class )(.*)(\()", line).group(2)
            if level == 0:
                module.addthing(name)
                classfile = class_file(package+"."+module_name, name, outpath)
                classfile.flush()
            print 4*level*" " + name,level
            
        if re.match("\s*def .*",line) is not None:
            level = len(re.split("d+", line)[0])
            if level > 0:
                if indent is None:
                    indent = level
                level = level / indent
            name  = re.search("(def )(.*)(\()", line).group(2)
            print 4*level*" " + name, level
            if level == 0:
                module.addthing(name)
                classfile = func_file(package+"."+module_name, name, outpath)
                classfile.flush()            

            
    f.close()
    module.flush()

        
if __name__ == "__main__":
    inpath   = os.path.abspath("./pyvsim")
    outpath  = os.path.abspath("./docsource")
    forbidden = ["couchdb","Oldversion","source",".git","build","examples"]
    xupadir(inpath, outpath, forbidden,"pyvsim")     
    