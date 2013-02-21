from __future__ import division
import numpy as np
import copy

def normalize(v):
    if v.ndim == 1:
        return v / np.dot(v,v)**0.5
    if v.ndim == 2:
        #nrows = np.size(v,0)
        ncols = np.size(v,1)
        norms = np.sum(v*v,1)**0.5
        norms = np.tile(norms.T,(ncols,1)).T
        return v / norms

# V1 = [[0,1,0],
      # [2,0,0],
      # [0,1,0],
      # [0,3,0.]]
# V1 = np.array(V1)

# V2 = [[0,0,1],
      # [0,0,1],
      # [1,0,0],
      # [1,0,0.]]
# V2 = np.array(V2)

# Pt = [[0,0,0],
      # [0,0,0],
      # [0,0,0],
      # [0,0,1.]]
# Pt = np.array(Pt) 
      
# N               = np.cross(V1,V2)
# UU              = np.sum(V1*V1,1)
# UV              = np.sum(V1*V2,1)
# VV              = np.sum(V2*V2,1)
# UVden           = (UV**2 - UU*VV)
      
      
# V = [[0,0,3],
     # [0,3,0.],
     # [1,0,0]]
# V = np.array(V)

# p0 = [[0.1,.05,-0.5],
      # [0.1,-0.1,0.2],
      # [-1,-1,0]]
# p0 = np.array(p0)

# ntris = np.size(V1,0)
# ndims = 3
# nlins = np.size(p0,0)

# #N = N.reshape(ntris,1,ndims)
# print "Normals"
# print N
# print "NNorms"
# print normalize(N)

# V = V.reshape(nlins,1,ndims)
# print "V"
# print V

# print "Mult"
# print N*V

# print "Den"
# den = np.sum(N*V,2)
# den[(den == 0)] = np.inf
# print den

# print "V0"
# P0 = p0.reshape(nlins,1,ndims)
# V0 = Pt - P0
# print V0

# print "Num"
# num = np.sum(N*V0,2)
# print num

# print "T"
# T  = num / den 
# print T

# T_0 = copy.deepcopy(T)
# T.resize(nlins,ntris,1)
# np.tile(T.T,(1,1,3))
# print "Reshaped T"
# print T
# print "Reshaped V"
# print V
# print "Intersections"
# print P0 + T * V
# P = P0 + T * V

# U       = P - Pt
# #UW      = vec.dot(V1,V0)
# UW      = np.sum(V1 * U,2)
# #VW      = vec.dot(V2,V0)  
# VW      = np.sum(V2 * U,2)

# print "UVden"
# print UVden
# S1 = (UV*VW - VV*UW) / UVden
# print "Intersections"
# print P0 + T * V

# print "Told"
# print T_0
# print "S1"
# print S1
# T1 = (UV*UW - UU*VW) / UVden
# print "T1"
# print T1

# tol = 1e-7
# [Ii,Ij] = np.nonzero((T_0 <= tol)+(T_0 > 1+tol)+ \
                      # (S1 < -tol)+(T1 < -tol)+ \
                      # (S1 + T1 > 1 + tol))
# print "Tindexes"
# print [Ii,Ij]

# T_0[Ii,Ij] = np.inf
# print T_0

# inds = np.argmin(T_0,1)
# print np.argmin(T_0,1)
# print T_0[range(nlins),inds]

# triangleIndexes         = np.argmin(T_0,1)
# lineParameters          = T_0[range(nlins),triangleIndexes]
# intersectionCoords      = P[range(nlins),triangleIndexes,:]
# triangleNormals         = N[triangleIndexes]

# print "triangleIndexes"
# print triangleIndexes
# print "lineParameters"
# print lineParameters
# print "intersectionCoords"
# print intersectionCoords
# print "triangleNormals"
# print triangleNormals

bounds = [[0,0,0],[1,1,1]]

p0 = np.array([[-0.1,0.5,0.5],
               [1.1,0.5,0.5],
               [-1,-1,-1],
               [0,0.00001,0],
               [-0.0001,0,0],])
               
p1 = np.array([[1.1,0.5,0.5],
               [1.2,0.5,0.5],
               [2999,2999,2999],
               [0.0001,0,0.0001],
               [-0.001,1,0],])

[xmin,xmax] =  bounds

V = p1 - p0
V[V == 0] = 1e-8

T1 = (xmin - p0) / V
#T1[T1 == np.inf] = 9
#T1[T1 == -np.inf] = -9
T2 = (xmax - p0) / V
#T2[T2 == np.inf] = 9
#T2[T2 == -np.inf] = -9

print "xmin - p0"
print (xmin - p0)
print "V"
print V
print "T1"
print T1
print "T2"
print T2

Tmin = T1 * (V >= 0) + T2 * (V < 0)
Tmax = T2 * (V >= 0) + T1 * (V < 0)

print "Tmin"
print Tmin
print "Tmax"
print Tmax

eliminated1 = (Tmin[:,0] > Tmax[:,1]) + (Tmin[:,1] > Tmax[:,0])
print "Eliminated1"
print eliminated1

Tmin[:,0] = Tmin[:,0] * (Tmin[:,1] <= Tmin[:,0]) + \
            Tmin[:,1] * (Tmin[:,1] > Tmin[:,0])
            
Tmax[:,0] = Tmax[:,0] * (Tmax[:,1] >= Tmax[:,0]) + \
            Tmax[:,1] * (Tmax[:,1] < Tmax[:,0])
            
eliminated2 = (Tmin[:,0] > Tmax[:,2]) + (Tmin[:,2] > Tmax[:,0])
print "Eliminated2"
print eliminated2

Tmin[:,0] = Tmin[:,0] * (Tmin[:,2] <= Tmin[:,0]) + \
            Tmin[:,2] * (Tmin[:,2] > Tmin[:,0])
            
Tmax[:,0] = Tmax[:,0] * (Tmax[:,2] >= Tmax[:,0]) + \
            Tmax[:,2] * (Tmax[:,2] < Tmax[:,0])
            
eliminated3 = (Tmin[:,0] > 1) + (Tmax[:,0] < 0)
print "Eliminated3"
print eliminated3

print "Grand result"
print 1 - (eliminated1 + eliminated2 + eliminated3)

# tmin = [0,0,0]
# tmax = [0,0,0]

# for d in range(3):
    # if (p1[d]-p0[d]) == 0.0:
        # divx = np.inf
    # else:
        # divx = 1.0 / (p1[d]-p0[d])
        
    # if (p1[d]-p0[d] >= 0):
        # tmin[d] = (xmin[d] - p0[d])*divx
        # tmax[d] = (xmax[d] - p0[d])*divx
    # else:
        # tmin[d] = (xmax[d] - p0[d])*divx
        # tmax[d] = (xmin[d] - p0[d])*divx

    
# if ((tmin[0] > tmax[1]) or (tmin[1] > tmax[0])):
    # return 0
    
# if (tmin[1] > tmin[0]):
    # tmin[0] = tmin[1]
# if (tmax[1] < tmax[0]):
    # tmax[0] = tmax[1] 
    
# if ((tmin[0] > tmax[2]) or (tmin[2] > tmax[0])):
    # return 0
    
# if (tmin[2] > tmin[0]):
    # tmin[0] = tmin[2]
# if (tmax[2] < tmax[0]):
    # tmax[0] = tmax[2]
    
# return ((tmin[0] < 1) * (tmax[0] > 0))   

#=========================
# Code vectorization test
#=========================
import numpy as np
class test(object):
    def __init__(self):
        self.myProp = 1
    def getMyProp(self, n):
        return self.myProp + n
        
a = test(); b = test(); c = test()
a.myProp = 1; b.myProp = 2; c.myProp = 3
l  = np.array([a,b,c])
l2 = np.array([1,2,3])
vfunc = np.vectorize(test.getMyProp)
vfunc(l,l2)

#====================================
#  Serialization implementation test
#====================================
import json
#---------------------
#  TOXIC CODE!!!
#---------------------
#
class NumpyEncoder(json.JSONEncoder):
    """A JSON encoder for numpy arrays
    Use like json.dumps(data, cls=NumpyEncoder)"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def numpy_decoder(dct):
    """Decodes numpy arrays stored as values in a json dictionary
    Use like json.loads(j, object_hook=numpy_decoder)"""
    for k in dct.keys():
        if isinstance(dct[k], list):
            try:
                dct[k] = np.asarray(dct[k])
            except ValueError:
                print "Couldn't convert to numpy :", dct[k]
                pass # can't convert to numpy array so leave as is
    return dct
#
#
#

import pickle
print pickle.dumps(assembly)
import json
enc = Utils.NumpyEncoder(indent=4)
dec = json.JSONDecoder()
code = enc.encode(part.__dict__)
part2 = Part()
decode = Utils.numpy_decoder(dec.decode(code))
part2.__dict__ = decode

print part2.x
print part2.y
print part2.z
print type(assembly) is Component
print type(part) is Component
print type(assembly) is Assembly
print isinstance(part, Component)