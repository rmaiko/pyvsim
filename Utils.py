#!/usr/bin/env python
"""
PyVSim part2.1
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
from __future__ import division
import numpy as np
# import copy
# import vtk
# import Object
# import pprint
import time
# import threading

HEXA_CONNECTIVITY = np.array([[0,1,4],
                             [1,5,4],
                             [4,5,6],
                             [4,6,7],
                             [2,6,7],
                             [2,7,3],
                             [0,2,3],
                             [0,1,2],
                             [0,7,3],
                             [0,4,7],
                             [1,5,6],
                             [1,6,2]])

HEXA_FACES_PER_NODE = np.array([[5,1,2],
                                  [4,1,2],
                                  [1,4,0],
                                  [1,5,0],
                                  [5,2,3],
                                  [2,4,3],
                                  [4,0,3],
                                  [5,3,0]])

# Definition of triangles using the hexa points
HEXA_CONN_PARTIAL = np.array([[1,4,0],
                            [3,1,0],
                            [4,3,0],
                            [5,2,6],
                            [7,5,6],
                            [2,7,6]])



def hexaInterpolation(p, pHexa, part2):
    """
    Interpolates values in an hexahedron.
    
    Vectorization here is totally ad-hoc, use with care
    """
    if p.ndim == 2:
        result = []
        for pint in p:
            result.append(hexaInterpolation(pint, pHexa, part2))
        return result
    else:
        p1 = np.tile(p,(12,1))
        p2 = pHexa[HEXA_CONNECTIVITY[:,0]]
        p3 = pHexa[HEXA_CONNECTIVITY[:,1]]
        p4 = pHexa[HEXA_CONNECTIVITY[:,2]]
        tetraVols = tetraVolume(p1,p2,p3,p4)
        
        Vp = np.empty(6)
        k = 0
        for n in range(0,12,2):
            Vp[k] = (tetraVols[n] + tetraVols[n+1])
            k = k + 1
            
        den = (Vp[0] + Vp[2]) * (Vp[1] + Vp[3]) * (Vp[5] + Vp[4])
    
        C = np.prod(Vp[HEXA_FACES_PER_NODE],1) / (den)
        
        if part2.ndim == 1:
            return np.sum(C * part2)
        else:
            return np.sum(np.tile(C,(np.size(part2,1),1)).T * part2, 0)
    
    
    
def tetraVolume(p1,p2,p3,p4):
    """
    Only for vectors
    """
    part2 = np.array([p1-p4, p2-p4, p3-p4])
    return (1/6)* np.abs(part2[0,:,0]*part2[1,:,1]*part2[2,:,2] + \
                         part2[2,:,0]*part2[0,:,1]*part2[1,:,2] + \
                         part2[1,:,0]*part2[2,:,1]*part2[0,:,2] - \
                         part2[2,:,0]*part2[1,:,1]*part2[0,:,2] - \
                         part2[0,:,0]*part2[2,:,1]*part2[1,:,2] - \
                         part2[1,:,0]*part2[0,:,1]*part2[2,:,2])

def metersToRGB(wl):
    gamma = 0.8
    wl = wl * 1e9
    f = (wl >= 380) * (wl < 420) * (0.3 + 0.7 * (wl - 380) / (420 - 380)) + \
        (wl >= 420) * (wl < 700) * 1 + \
        (wl >= 700) * (wl < 780) * (1 - 0.7 * (wl - 700) / (780 - 700))
    
    r = (wl >= 380) * (wl < 440) * (1 - (wl - 380) / (440 - 380)) + \
        (wl >= 440) * (wl < 510) * 0 + \
        (wl >= 510) * (wl < 580) * ((wl - 510) / (580 - 510)) + \
        (wl >= 580) * (wl < 780) * 1
    
    g = (wl >= 380) * (wl < 440) * 0 + \
        (wl >= 440) * (wl < 490) * ((wl - 440) / (490 - 440)) + \
        (wl >= 490) * (wl < 580) * 1 + \
        (wl >= 580) * (wl < 645) * (1 - (wl - 580) / (645 - 580)) + \
        (wl >= 645) * (wl < 780) * 0
        
    b = (wl >= 380) * (wl < 490) * 1 + \
        (wl >= 490) * (wl < 510) * (1 - (wl - 490) / (510 - 490)) + \
        (wl >= 510) * (wl < 780) * 0
        
    return ((np.array([r,g,b])*f)**gamma)
    

def aeq(a,b,tol=1e-8):
    """
    A tool for comparing numpy nparrays and checking if all their values
    are Almost EQual up to a given tolerance::
    
    >>> aeq(0,0.00000000000001)
    True
    >>> aeq(np.array([1,0,0]),np.array([0.99999999999999,0,0]))
    True
    >>> aeq(np.array([1,0,0]),np.array([0.99999,0,0]))
    False
    """
    temp = np.abs(a - b) > tol
    try:
        if len(temp[temp == True]) > 0:
            return False
        else:
            return True
    except:
        return not temp

class Tictoc:
    """
    Just something simpler than the timeit from Python 
    (and more Matlab-style)
    """
    def __init__(self):
        """
        This is the class constructor
        >>> tic = Tictoc()
        """
        self.begin = 0
        
    def tic(self):
        """
        Resets the timer, must be used at the before each timed method
        >>> tic = Tictoc()
        >>> tic.tic()
        """
        self.begin = time.clock()
        
    def toc(self,n=None):
        """
        Gives the calculation time or speed, depending if the number of 
        calculations executed from the last reset is provided as the 
        input n.
        
        >>> tic = Tictoc()
        >>> tic.tic()
        >>> tic.toc() # doctest: +ELLIPSIS
        Elapsed time: ...
        ...
        
        or
        
        >>> tic = Tictoc()
        >>> tic.tic()
        >>> tic.toc(10) # doctest: +ELLIPSIS
        Can execute: ... calculations / second
        ...
        
        """
        t = (time.clock()-self.begin)
        if n is None:
            print "Elapsed time: %f seconds" % t
            return t
        else:
            print "Can execute: %f calculations / second" % (n/t)
            return n/t
               
def rotateVector(x,angle,axis):
    """
    This implementation uses angles in degrees. The algorithm is the vectorized
    formulation of the `Euler-Rodrigues formula 
    <http://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_parameters>`_
    
    x
        A vector (size = 3) or a list of vectors (n rows, 3 columns) to be
        rotated
    angle
        scalar (in radians)
    axis
        A vector around which the vector is rotated
        
    For example::
    
    >>> [x,y,z] = np.eye(3)
    >>> temp = rotateVector(x, np.pi/2, z)
    >>> aeq(temp, y)
    True
    
    This works also for lists of vectors::
    
    >>> X = np.tile(x,(100,1))
    >>> Y = np.tile(y,(100,1))
    >>> Z = np.tile(z,(100,1))
    >>> temp = rotateVector(X, np.pi/2, Z)
    >>> aeq(temp, Y)
    True
    """
    a = np.cos(angle/2)
    w = axis*np.sin(angle/2)
    return x + 2*a*np.cross(w,x) + 2*np.cross(w,np.cross(w,x))
    
def rotatePoints(points,angle,axis,origin):
    """
    Wrap-around `Euler-Rodrigues formula 
    <http://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_parameters>`_
    formula for rotating a point cloud
    
    points
        A point (size = 3) or a list of points (n rows, 3 columns) to be
        rotated
    angle
        scalar (in radians)
    axis
        A vector around which the points are rotated
    origin
        A point around which the points are rotated
        
    For example::
    
    >>> o = np.array([0,0,0])
    >>> [x,y,z] = np.eye(3)
    >>> temp = rotatePoints(x, np.pi/2, z, o)
    >>> aeq(temp, y)
    True
    
    This works also for lists of vectors::
    
    >>> X = np.tile(x,(100,1))
    >>> Y = np.tile(y,(100,1))
    >>> Z = np.tile(z,(100,1))
    >>> temp = rotatePoints(X, np.pi/2, Z, o)
    >>> aeq(temp, Y)
    True
    """
    if len(points) > 0:
        return rotateVector(points-origin,angle,axis) + origin
    else:
        return points
        
def normalize(part2):
    """
    This can be used to normalize a vector or a list of vectors, provided
    they are given as numpy arrays.
    
    >>> result = normalize(np.array([2,0,0]))
    >>> assert(aeq(result, np.array([1,0,0])))
    
    And for multiple vectors:
    
    >>> result = normalize(np.array([[2,0,0], [0,0,1]]))
    >>> assert(aeq(result, np.array([[1,0,0], [0,0,1]])))
    
    """
    if part2.ndim == 1:
        return part2 / np.dot(part2,part2)**0.5
    if part2.ndim == 2:
        ncols = np.size(part2,1)
        norms = norm(part2)
        norms[norms == 0] = 1
        norms = np.tile(norms.T,(ncols,1)).T
        return part2 / norms
        
def norm(part2):
    """
    This can be used to calculate the euclidean norm of vector or a list of 
    vectors, provided they are given as numpy arrays of the following form::
    
    >>> inp = norm(np.array([1,1,0]))
    >>> out = np.sqrt(2)
    >>> assert(aeq(inp, out))
    
    And for multiple vectors:
    
    >>> inp = norm(np.array([[1,1,0],[1,1,1]]))
    >>> out = np.array([np.sqrt(2), np.sqrt(3)])
    >>> assert(aeq(inp, out))
    """
    if part2.ndim == 1:
        return np.dot(part2,part2)**0.5
    if part2.ndim == 2:
        return np.sum(part2*part2,1)**0.5
       
def barycentricCoordinates(p,p1,p2,p3):
    """
    Returns the barycentric coordinates of points, given the triangles where
    they belong. 
    
    For more information on barycentric coordinates, see `Wikipedia 
    <http://en.wikipedia.org/wiki/Barycentric_coordinate_system>`
    
    Inputs::
    p
        point in space, or a list of points in space
    p1,p2,p3 
        points space representing triangle (can be lists)
    
    Outputs::
    [lambda1,lambda2,lambda3]
        the barycentric coordinates of point p with respect to the defined triangle
    
    Assumes::
    1) points are given as numpy arrays (crashes if not met)
    
    Algorithm::
        area        = Object.triangleArea(p1,p2,p3)
        lambda1     = Object.triangleArea(p,p2,p3)
        lambda2     = Object.triangleArea(p,p1,p3)
        lambda3     = Object.triangleArea(p,p1,p2)
        return np.array([lambda1,lambda2,lambda3])/area
    """
    area        = triangleArea(p1,p2,p3)
    lambda1     = triangleArea(p,p2,p3)
    lambda2     = triangleArea(p,p1,p3)
    lambda3     = triangleArea(p,p1,p2)
    if p1.ndim == 1:
        return np.array([lambda1,lambda2,lambda3])/area
    if p1.ndim == 2:
        return (np.array([lambda1,lambda2,lambda3]) / area).T       
   
def triangleArea(p1,p2,p3):
    """
    Given three points in space, returns the triangle area. 
    
    Assumes::
    
    1) points are given as numpy arrays (crashes if not met)
    2) if points lists are given, this will still work
    
    Algorithm:
        v1 = p2 - p1
        v2 = p3 - p1
        return 0.5*norm(np.cross(v1,v2))
    """ 
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5*norm(np.cross(v1,v2))
            

def readSTL(filename):
    import vtk
    import Core
    STLReader   = vtk.vtkSTLReader()

    STLReader.SetFileName(filename)
    STLReader.Update()
    polydata = STLReader.GetOutput()    
    points      = polydata.GetPoints()
   
    pts = []
    cts = []
    for n in range(points.GetNumberOfPoints()):
        pts.append(points.GetPoint(n))
   
    for n in range(polydata.GetNumberOfCells()):
        temp = vtk.vtkIdList()
        polydata.GetCellPoints(n,temp)
        cts.append([temp.GetId(0),temp.GetId(1),temp.GetId(2)])
       
    obj                 = Core.Part()
    obj.points          = np.array(pts)
    obj.connectivity    = np.array(cts)
   
    return obj
    
def pointInHexa(p,hexapoints):
    """
    Taking a set of points defining a hexahedron in the conventional order,
    this function tests of the point is inside this hexahedron by:
   
    For each face:
        - calculate normal pointing outwards
        - verify if point is "behind" the plane defined by the face
       
    Returns 1 if inside or on the face, 0 otherwise
    """

   
    if p.ndim > 1:
        truthtable = np.ones(len(p))
    else:
        truthtable = 1
              
    for n in range(6):
        vout = np.cross(hexapoints[HEXA_CONN_PARTIAL[n,0]]-hexapoints[HEXA_CONN_PARTIAL[n,2]],
                        hexapoints[HEXA_CONN_PARTIAL[n,1]]-hexapoints[HEXA_CONN_PARTIAL[n,2]])
        truthtable = truthtable * (listdot(p - hexapoints[HEXA_CONN_PARTIAL[n,2]],vout) < 0)

    return truthtable        
   
def listdot(a,b):
    # If numpy dot product is capable of doing the job
    if (a.ndim == b.ndim) and (a.ndim == 1):
        return np.dot(a,b)
    else:
        # Helps eliminating further ifs
        if a.ndim < b.ndim:
            small = a
            large = b
        else:
            small = b
            large = a
        # Dot product of a vector and a list of vectors
        if (small.ndim == 1) and (len(small) == len(large[0])):
            return np.sum(np.tile(small,(len(large),1)) * large,1)
        # Dot product of two list of vectors:
        if (small.ndim == large.ndim) and (np.size(small) == np.size(large)):
            return np.sum(a*b,1)
           
def listTimesVec(HEXA_CONN_PARTIAL,part2):
    try:
        if part2.ndim > 1:
            return np.tile(HEXA_CONN_PARTIAL,(len(part2[0]),1)).T * part2
        else:
            return np.tile(HEXA_CONN_PARTIAL,(len(part2),1)).T * np.tile(part2,(len(HEXA_CONN_PARTIAL),1))
    except:
        return HEXA_CONN_PARTIAL*part2
       
def quadInterpolation(p,p1,p2,p3,p4,v1,v2,v3,v4): 
    """
    Performs barycentric interpolation extended to the case of a planar quad
    """
    Su = triangleArea(p,p4,p3)
    Sr = triangleArea(p,p3,p2)
    Sd = triangleArea(p,p1,p2)
    Sl = triangleArea(p,p1,p4)
    den = (Su + Sd)*(Sr + Sl)
    c1 = Sr*Su/den
    c2 = Sl*Su/den
    c3 = Sl*Sd/den
    c4 = Sr*Sd/den
    return listTimesVec(c1,v1) + listTimesVec(c2,v2) + \
           listTimesVec(c3,v3) + listTimesVec(c4,v4)
           
# def quadArea(p1,p2,p3,p4):
    # return triangleArea(p1,p2,p3) + triangleArea(p1,p3,p4)
       
# def triangleNormal(p1,p2,p3):
    # """
    # Given three points in space, returns the triangle normal. Assumes:
    # 1 - points are given as numpy arrays (crashes if not met)
    # 2 - points are given counterclockwise (negative result if not met)
    
    # Algorithm:
        # v1 = p2 - p1
        # v2 = p3 - p1
        # N = np.cross(v1,v2)
        # return N/np.linalg.norm(N)
    # """
    # v1 = p2 - p1
    # v2 = p3 - p1
    # N = np.cross(v1,v2)
    # return vec.normalize(N)

    

#points = [[0,0,0],
#          [1,0,0],
#          [1,1,0],
#          [0,1,0],
#          [0,0,1],
#          [1,0,1],
#          [1,1,1],
#          [0,1,1]]
#    
#points = np.array(points)
#p = np.array([0.5,1,0.5])
#part2 = np.array([0,0,0,0,1,1,1,1])
##part2 = np.array([[0,0,0],[0,0,0],[1,0,0],[1,0,0],[0,0,1],[0,0,1],[0,1,1],[0,1,1]])
#print hexaInterpolation(p, points, part2)
#
#p1 = np.array([0,0,0])
#p2 = np.array([1,0,0])
#p3 = np.array([0,1,0])
#p4 = np.array([0,0,1])
#
#reps = 100000
#volumedet = np.zeros(reps)
#volumevec = np.zeros(reps)
#
##
## Determinant method
##
#tic = Tictoc()
#tic.tic()
#for n in range(reps):
#    D = np.ones((4,4))
#    D[0,1:4] = p1
#    D[1,1:4] = p2
#    D[2,1:4] = p3
#    D[3,1:4] = p4
#    volumedet[n] = (1/6)*np.linalg.det(D)
#tic.toc(reps)
#
##
## Vector method
##
#P1 = np.tile(p1,(reps,1))
#P2 = np.tile(p2,(reps,1))
#P3 = np.tile(p3,(reps,1))
#P4 = np.tile(p4,(reps,1))
#
#tic.tic()
#volumevec = tetraVolume(P1,P2,P3,P4)
#tic.toc(reps)
#
#assert (volumevec == volumedet).all()