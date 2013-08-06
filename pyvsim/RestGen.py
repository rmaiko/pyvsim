'''
Created on 6 aout 2013

@author: maiko
'''
import re

class filenames(object):
    def __init__(self):
        self.level = 0
        self.file  = None
        self.path  = "./temp/"
        
    def addthing(self, level, thing, t):
        if level < self.level:
            self.file.close()
            self.file = None

        if level == 0:
            self.file = open(self.path + thing + ".rst", "w")
            self.file.write(thing + "\n")
            self.file.write("*"*len(thing)+"\n")
            if t == 0:
                self.file.write(".. autoclass:: "+thing+"\n")
            if t == 1:
                self.file.write(".. autofunction:: "+thing+"\n")
        else:
            pass
            
    def flush(self):
        if self.file is not None:
            self.file.close()

def chupatudo(filename):
    f = open(filename, "r")
    indent = None
    struct = filenames()
    
    for line in f:
        if re.match("\s*class .*",line) is not None:
            level = len(re.split("c+", line)[0])
            if level > 0:
                if indent is None:
                    indent = level
                level = level / indent
            name  = re.search("(class )(.*)(\()", line).group(2)
            struct.addthing(level, name,0)
            print 4*level*" ", name 
            
        if re.match("\s*def .*",line) is not None:
            level = len(re.split("d+", line)[0])
            if level > 0:
                if indent is None:
                    indent = level
                level = level / indent
            name  = re.search("(def )(.*)(\()", line).group(2)
            print 4*level*" ", name 
            struct.addthing(level, name,1)
            
    f.close()
    struct.flush()

if __name__ == "__main__":
    filename = "Utils.py"
    chupatudo(filename)
    