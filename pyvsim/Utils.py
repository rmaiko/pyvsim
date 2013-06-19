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
from __future__ import division
import numpy as np
import scipy.linalg
import time
import warnings
import ConfigParser

CONFIG_FILE = "./config.dat"

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

def readConfig(section, field):
    parser = ConfigParser.SafeConfigParser()
    parser.read(CONFIG_FILE)
    return parser.get(section, field)

def hexaInterpolation(p, hexapoints, values):
    """
    This function interpolates values given on vertices of an hexahedron to 
    a point. The algorithm relies on the ratio of volumes of tetrahedrons
    defined by the hexa faces and the point, and its behavior is approximately
    linear.
    
    *Warning* - there is no verification whether the points are inside the 
    hexahedron. One must check it using other methods.
    
    Parameters
    ----------
    p : numpy.array (N, 3)
        List of points to be interpolated
    hexapoints : numpy.array (8, 3)
        List of points defining the hexahedron
    values : numpy.array (M, 8)
        List of values at the vertices of the hexahedron
        
    Returns
    -------
    interpolated : numpy.array (N, M)
        Values interpolated at points p
    """
    assert (hexapoints.shape == (8,3))
    assert np.size(values,0) == 8
    
    if p.ndim == 1:
        # This is done if only one point is given
        p = np.reshape(p,(1,np.size(p)))

    # Creates the list of coordinates of the tetrahedrons
    p1 = np.reshape(p,(np.size(p,0),1,np.size(p,1)))
    p2 = [hexapoints[HEXA_CONNECTIVITY[:,0]]]
    p3 = [hexapoints[HEXA_CONNECTIVITY[:,1]]]
    p4 = [hexapoints[HEXA_CONNECTIVITY[:,2]]]

    # The lists are repeated to match dimensions
    # Notice that there are 12, as it's not possible to calculate the volume of
    # a tetra with square base
    p1 = np.tile(p1,(1,12,1))
    p2 = np.tile(p2,(np.size(p,0),1,1))
    p3 = np.tile(p3,(np.size(p,0),1,1))
    p4 = np.tile(p4,(np.size(p,0),1,1))
    
    # Calculate the volumes
    tetraVols = tetraVolume(p1,p2,p3,p4)
    
    # Allocates list to store the tetra volumes
    # Notice there are only 6 slots, so we have to add the areas
    Vp = np.empty((np.size(p,0),6))

    # We have to sum the areas to obtain the volume of tetras with quad base
    k = 0
    for n in range(0,12,2):
        Vp[:,k] = (tetraVols[:,n] + tetraVols[:,n+1])
        k = k + 1

    # Takes the area of opposite hexahedrons
    den = (Vp[:,0] + Vp[:,2]) * (Vp[:,1] + Vp[:,3]) * (Vp[:,5] + Vp[:,4])

    # Now we take the corresponding volumes per node and divide by the factor
    # calculated above. This yields the weights to average the values
    C = np.prod(Vp[:,HEXA_FACES_PER_NODE],2) / np.reshape(den,(np.size(p,0),1))

    if values.ndim == 1:
        return np.sum(C * values,1).squeeze()
    else:
        C = np.reshape(C,(np.size(C,0),1,8))
        C = np.tile(C, (1,np.size(values,1),1))
        values = np.array([values.T])
        return np.sum(C * values,2).squeeze()

    
    
    
def tetraVolume(p1,p2,p3,p4):
    """
    Calculates the volume of a tetrahedron. This is simply the unrolled
    determinant::
    
    | Vx1  Vy1  Vz1 |     1
    | Vx2  Vy2  Vz2 | * -----
    | Vx3  Vy3  Vz3 |     6
    
    
    This function works only for list of vectors, for performance reasons
    will not check the inputs, will throw an error instead.
    
    (This works faster than numpy.linalg.det repeated over the list
    """
    vecs = np.array([p1-p4, p2-p4, p3-p4])
    if p1.ndim == 3:
        return (1/6)* np.abs(vecs[0,:,:,0]*vecs[1,:,:,1]*vecs[2,:,:,2] +
                             vecs[2,:,:,0]*vecs[0,:,:,1]*vecs[1,:,:,2] +
                             vecs[1,:,:,0]*vecs[2,:,:,1]*vecs[0,:,:,2] -
                             vecs[2,:,:,0]*vecs[1,:,:,1]*vecs[0,:,:,2] -
                             vecs[0,:,:,0]*vecs[2,:,:,1]*vecs[1,:,:,2] -
                             vecs[1,:,:,0]*vecs[0,:,:,1]*vecs[2,:,:,2])
    else:
        return (1/6)* np.abs(vecs[0,:,0]*vecs[1,:,1]*vecs[2,:,2] +
                             vecs[2,:,0]*vecs[0,:,1]*vecs[1,:,2] +
                             vecs[1,:,0]*vecs[2,:,1]*vecs[0,:,2] -
                             vecs[2,:,0]*vecs[1,:,1]*vecs[0,:,2] -
                             vecs[0,:,0]*vecs[2,:,1]*vecs[1,:,2] -
                             vecs[1,:,0]*vecs[0,:,1]*vecs[2,:,2]) 

def jet(value, minval, maxval):
    """
    Returns the RGB values to emulate a "jet" colormap from matlab
    """
    val = 4 * (value - minval)/(maxval-minval)
    rmask = val - 1.5 < -val + 4.5
    gmask = val - 0.5 < -val + 3.5
    bmask = val + 0.5 < -val + 2.5
    r   = (val - 1.5)*rmask + (-val + 4.5)*(1-rmask)
    g   = (val - 0.5)*gmask + (-val + 3.5)*(1-gmask)
    b   = (val + 0.5)*bmask + (-val + 2.5)*(1-bmask)
    return np.clip(np.vstack([r,g,b]).T, 0, 1)

def metersToRGB(wl):
    """
    Converts light wavelength to a RGB vector, the algorithm comes from:
    `This blog <http://codingmess.blogspot.de/2009/05/conversion-of-wavelength-in-nanometers.html>`
    
    Parameters
    ----------
    wl : scalar
        The wavelength in meters
        
    Returns
    -------
    [R,G,B] : numpy.array (3)
        The normalized (0..1) RGB value for this wavelength
    """
    gamma = 0.8
    wl = wl * 1e9
    f =((wl >= 380) * (wl < 420) * (0.3 + 0.7 * (wl - 380) / (420 - 380)) + 
        (wl >= 420) * (wl < 700) * 1 + 
        (wl >= 700) * (wl < 780) * (1 - 0.7 * (wl - 700) / (780 - 700)))
        
    r =((wl < 475)  * (1 - (wl - 380) / (440 - 380)) + 
        (wl >= 475) * ((wl - 510) / (580 - 510)))
    
    g =((wl < 535) * ((wl - 440) / (490 - 440)) + 
        (wl >= 535)* (1 - (wl - 580) / (645 - 580)))
    
    b =(1 - (wl - 490) / (510 - 490))
    
    return (((np.clip([r,g,b],0,1))*f)**gamma)

def aeq(a,b,tol=1e-8):
    """
    A tool for comparing numpy nparrays and checking if all their values
    are *A*lmost *EQ*ual up to a given tolerance.
    
    Parameters
    ----------
    a,b : numpy.arrays or scalars
        Values to be compared, must have the same size
    tol = 1e-8
        The tolerance of the comparison
        
    Returns
    -------
    True, False
        Depending if the absolute difference between of one of the values of
        the inputs exceeds the tolerance
    
    Examples
    --------
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
    except IndexError:
        return not temp
       
def reallocateArray(array, extrasteps):
    size = [np.size(array,0) + extrasteps]
    for n in range(1,array.ndim):
        size.append(np.size(array,n))
    temp = np.empty(size, dtype = array.dtype)
    temp[range(np.size(array,0))] = array
    return temp
               
def rotateVector(x,angle,axis):
    """
    This implementation uses angles in degrees. The algorithm is the vectorized
    formulation of the `Euler-Rodrigues formula 
    <http://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_parameters>`_
    
    Parameters
    ----------
    x : numpy.array (N, 3)
        A vector (size = 3) or a list of vectors (N rows, 3 columns) to be
        rotated
    angle : double
        scalar (in radians)
    axis : numpy.array (3)
        A vector around which the vector is rotated
        
    Returns
    -------
    vectors : numpy.array (N, 3)
        The vectors after rotation
        
    Examples
    --------
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
    
    Parameters
    ----------
    points : numpy.array (N, 3)
        A point (size = 3) or a list of points (N rows, 3 columns) to be
        rotated
    angle : scalar
        scalar (in radians)
    axis : numpy.array (3)
        A vector around which the points are rotated
    origin : numpy.array (3)
        A point around which the points are rotated
        
    Returns
    -------
    points : numpy.array (N, 3)
        A list of rotated points
        
    Examples
    --------    
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
        
def normalize(vectors):
    """
    This can be used to normalize a vector or a list of vectors, provided
    they are given as numpy arrays.
    
    Parameters
    ----------
    vectors : numpy.array (N, 3)
        A list of vectors to be normalized
        
    Returns
    -------
    vectors : numpy.array (N, 3)
        The normalized vectors
    
    Examples
    --------    
    >>> result = normalize(np.array([2,0,0]))
    >>> assert(aeq(result, np.array([1,0,0])))
    
    And for multiple vectors:
    
    >>> result = normalize(np.array([[2,0,0], 
    ...                              [0,0,1]]))
    >>> assert(aeq(result, np.array([[1,0,0], 
    ...                              [0,0,1]])))
    
    """
    if vectors.ndim == 1:
        return vectors / np.dot(vectors,vectors)**0.5
    if vectors.ndim == 2:
        ncols = np.size(vectors,1)
        norms = norm(vectors)
        norms[norms == 0] = 1
        norms = np.tile(norms.T,(ncols,1)).T
        return vectors / norms
        
def norm(vectors):
    """
    This can be used to calculate the euclidean norm of vector or a list of 
    vectors, provided they are given as numpy arrays.
    
    Parameters
    ----------
    vectors : numpy.array (N, 3)
        A list of vectors
        
    Returns
    -------
    norms : numpy.array (N)
        A list with the euclidean norm of the vectors
        
    Examples
    --------    
    >>> inp = norm(np.array([1,1,0]))
    >>> out = np.sqrt(2)
    >>> assert(aeq(inp, out))
    
    And for multiple vectors:
    
    >>> inp = norm(np.array([[1,1,0],
    ...                      [1,1,1]]))
    >>> out = np.array([np.sqrt(2), 
    ...                 np.sqrt(3)])
    >>> assert(aeq(inp, out))
    """
    if vectors.ndim == 1:
        return np.dot(vectors,vectors)**0.5
    if vectors.ndim == 2:
        return np.sum(vectors*vectors,1)**0.5
       
def barycentricCoordinates(p,p1,p2,p3):
    """
    Returns the barycentric coordinates of points, given the triangles where
    they belong. 
    
    For more information on barycentric coordinates, see `Wikipedia 
    <http://en.wikipedia.org/wiki/Barycentric_coordinate_system>`
    
    Assumes::
    1) points are given as numpy arrays (crashes if not met)
    
    Algorithm
    ---------
    area        = Object.triangleArea(p1,p2,p3)
    lambda1     = Object.triangleArea(p,p2,p3)
    lambda2     = Object.triangleArea(p,p1,p3)
    lambda3     = Object.triangleArea(p,p1,p2)
    return np.array([lambda1,lambda2,lambda3])/area
    
    Parameters
    ----------
    p
        point in space, or a list of points in space
    p1,p2,p3 
        points space representing triangle (can be lists)
    
    Returns
    -------
    [lambda1,lambda2,lambda3]
        the barycentric coordinates of point p with respect to the defined 
        triangle
        
    Examples
    --------   
    >>> [p,p1,p2,p3] = np.array([[0.5,0.5, 0],
    ...                          [  0,  0, 0],
    ...                          [  1,  0, 0],
    ...                          [  0,  1, 0]])
    >>> barycentricCoordinates(p,p1,p2,p3)
    array([ 0. ,  0.5,  0.5])
    
    Will also work for arrays::
    
    >>> p  = np.tile(p,(3,1));  p1 = np.tile(p1,(3,1))
    >>> p2 = np.tile(p2,(3,1)); p3 = np.tile(p3,(3,1))
    >>> barycentricCoordinates(p,p1,p2,p3)
    array([[ 0. ,  0.5,  0.5],
           [ 0. ,  0.5,  0.5],
           [ 0. ,  0.5,  0.5]])
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
    Given three points in 3D space, returns the triangle area. 
    
    Assumes::
    
    1. points are given as numpy arrays (crashes if not met)
    2. if points lists are given, this will still work
    
    Algorithm
    ---------
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5*norm(np.cross(v1,v2))
        
    Parameters
    ----------
    p1, p2, p3 : numpy.array
        If a list of points is given, they must be vertically stacked.
    
    Returns
    -------
    area : scalar / numpy.array
        The area of the points defined by the triangles. If lists of points
        were used as inputs, the output is a 1D numpy.array with as many
        elements as given points 
    """ 
    v1 = p2 - p1
    v2 = p3 - p1
    return 0.5*norm(np.cross(v1,v2))
      
def KQ(A):
    """
    This decomposition is proposed in the book "Multiple View Geometry in
    computer vision" by Hartley and Zisserman. It is basically a RQ 
    decomposition (which takes a matrix M and finds a right, upper diagonal
    camera matrix K and a orthogonal matrix Q so that M = a*K*Q (a is a 
    normalizing factor (R[-1,-1])).
    
    This specific function has the following extra steps: 
    
    1) it defines a diagonal matrix D which, when post-multiplied by K makes 
    its diagonal elements positive.
     
    2) it normalizes K by its [-1,-1] element.
    
    The use of these steps is that when the matrix M is a DLT matrix, K is a 
    camera matrix, and Q is the orientation of the camera (its rows are the
    front, down and left vectors, respectively).
    
    Attention : a check is performed to verify that the 
    
    Parameters
    ----------
    A : numpy.array
        A square matrix. *Attention*, DLT matrices need to have their last
        column taken away for this procedure.
    
    Returns
    -------
    K : numpy.array
        The camera matrix, normalized by its [-1,-1] element.
    Q : numpy.array
        The camera orientation matrix
    """
    R, Q = scipy.linalg.rq(A)
    D = np.diag(np.sign(np.diag(R)))
    K = np.dot(R,D)
    Q = np.dot(D,Q) #D^-1 = D     
#    print "K\n", K
#    print "D\n", D
#    print "Q\n", Q

    if np.max(np.abs(A[:,:3] - np.dot(K,Q))) > 1e-10:
        print "WARNING - KQ decomposition failed, residue: \n", A - np.dot(K,Q)
        
    return K / K[-1,-1], Q

def pointSegmentDistance(p1, p2, x):
    """
    Given one point and some line segments, calculates the euclidean distance
    between each segment and this point.
    
    If the point lies outside segment, returns the distance between point and
    nearest extremity of the segment    
    
    Parameters
    ----------
    p1 : numpy.array (N,3)
        Coordinates of segments' initial points
    p2 : numpy.array (N,3)
        Coordinates of segments' final points
    x : numpy.array(3)
        Coordinates of point
        
    Returns
    -------
    distance : numpy.array (N)
        Distance between each of the segments and the point
        
    Examples
    --------
    >>> p1 = np.array([[  0,  0, 0],
    ...                [  0,  0, 0],
    ...                [  0,  0, 0]])
    >>> p2 = np.array([[  1,  0, 0],
    ...                [  0,  1, 0],
    ...                [  0,  0, 1]])  
    >>> x  = np.array([  1,  1, 1])
    >>> pointSegmentDistance(p1, p2, x)
    array([ 1.41421356,  1.41421356,  1.41421356])
    >>> x  = np.array([  2,  0, 0])
    >>> pointSegmentDistance(p1, p2, x)
    array([ 1.,  2.,  2.])
    """
    v    = (p2 - p1) 
    vlen = np.sqrt(np.sum(v*v,1))
    if (vlen == 0).any():
        return np.ones_like(vlen)*1000
    v    = np.einsum("ij,i->ij", v, 1 / vlen)
    # Calculate vectors from segment extremities to point
    p1x = x - p1
    p2x = x - p2
    # Calculate point-line (not segment) distance
    d_v_x   = np.cross(p1x,v)
    d_v_x   = np.sqrt(np.sum(d_v_x*d_v_x,1))
    # Calculate the projection of the point p at the line
    p1_x_prime = np.abs(np.sum(p1x*v, 1))
    p2_x_prime = np.abs(np.sum(p2x*v, 1))
    insegment  = aeq(p1_x_prime + p2_x_prime, vlen) 
    # Calculate point-point distances
    d_p1_x = np.sqrt(np.sum(p1x*p1x,1))
    d_p2_x = np.sqrt(np.sum(p2x*p2x,1))
    d_extrem = np.min(np.vstack([d_p1_x,d_p2_x]),0)
    return d_v_x*insegment + d_extrem*(1 - insegment)

def linesIntersection(v,p):
    """
    Calculates the intersection of a list of lines. If no intersection exists,
    will return a point that minimizes the square of the distances to the
    given lines.
    
    Parameters
    ----------
    v : numpy.array (N, M)
        A list of vectors with the direction of the lines (for 3D vectors, 
        M = 3)
    p : numpy.array (N, M)
        A list of vectors with a point in the line
        
    Returns
    -------
    x : numpy.array (M)
        The point that minimizes the square of the distance to each line
        
    Raises
    ------
    numpy.linalg.LinAlgError
        If the given lines are almost parallel, so no unique solution can be
        found
        
    Examples
    --------
    >>> v = np.array([[1,0,0],
    ...               [0,1,0]])
    >>> p = np.array([[0,0,0],
    ...               [0,-1,0]])
    >>> linesIntersection(v,p)
    array([ 0.,  0.,  0.])
    
    >>> p = np.array([[0,0,0],[0,-1,0.1]]) # No intersection exists
    >>> linesIntersection(v,p)
    array([ 0.  ,  0.  ,  0.05])
    """
    v = normalize(v)
    nlines  = np.size(v,0)
    NN      = np.zeros((nlines,3,3))
    NNp     = np.zeros((nlines,3,1))
    for n in range(nlines):
        NN[n]  = np.eye(3) - np.dot(np.reshape(v[n],(3,1)), 
                                    np.reshape(v[n],(1,3)))
        NNp[n] = np.dot(NN[n], np.reshape(p[n],(3,1))) 
        
    part1 = np.sum(NN,  0)
    part2 = np.sum(NNp, 0)
        
    [U,D,V] = np.linalg.svd(part1)
    if D[0]/D[-1] > 1e10:
        raise np.linalg.LinAlgError("Could not converge calculation")
    
    part1 = np.dot(V.T, np.dot(np.diag(1/D), U.T))
    
    return np.dot(part1,part2).squeeze()

def readSTL(filename):
    import vtk
    import pyvsim.Primitives
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
       
    obj                 = pyvsim.Primitives.Part()
    obj.points          = np.array(pts)
    obj.connectivity    = np.array(cts)
    obj.name            = filename
    
    print "Read %i triangles" % np.size(cts,0)
    
    return obj

def DLT(uvlist, xyzlist):
    """
    This function calculates the direct linear transform matrix, which is the
    transform executed by a pinhole camera. This procedure was suggested in
    `Wikipedia <http://en.wikipedia.org/wiki/Direct_linear_transformation>` 
    and some refinements are discussed in the book "Multiple View Geometry in
    computer vision" by Hartley and Zisserman.
    
    Parameters
    ----------
    uvlist : numpy.array
        A (N,2) matrix containing points at the sensor coordinates
    xyzlist: numpy.array
        A (N,3) matrix containing points at the world coordinates
        
    Returns
    -------
    M : numpy.array (3x4)
        A matrix with the transformation to be used with homogeneous
        coordinates. The matrix M is normalized by the norm of the elements
        M(2,0:3), because then the depth of points is automatically given as
        the third (the homogeneous) coordinate.
    dMdX : numpy.array
        A matrix containing the factors to calculate the partial derivatives
        of the UV coordinates with respect to the XYZ coordinates. By 
        multiplying dMdX * M * XYZ1, one gets the derivatives in the following
        order [du/dx du/dy du/dz dv/dx dv/dy dv/dz] multiplied by w^2 (which,
        for the normalized matrix M, is the depth of the points)
    detM : scalar
        The determinant of the matrix formed by the first three columns of M,
        if det(M[:,:3]) < 0, it indicates that the mapping is done from a
        right-handed coordinate system to a left-handed one (or vice versa). 
        This case happens when doing a mapping with a odd number of mirrors
        between camera and mapped region, and some derived quantities must be
        inverted in this case, e.g. the line-of-sight vector.
    condition_number : double
        The condition number stated in page 108 of Hartley and Zisseman, which
        is the ratio of the first and the second-last singular value (because
        the last should be zero, if the transform is perfect. According to
        `Wikipedia <http://en.wikipedia.org/wiki/Condition_number>`, the 
        log10 of the condition number gives roughly how many digits of 
        accuracy are lost by transforming using the given matrix.
    last_singular_value: double
        The smallest singular value. The finding of the DLT matrix is a 
        minimization of the problem abs(A*x) with abs(x) = 1. 
        last_condition_number is exactly abs(A*x), and gives an idea of the
        precision of the matrix found (with 0 being perfect)
    """
    assert np.size(uvlist,0)  == np.size(xyzlist,0)
    assert np.size(uvlist,1)  == 2
    assert np.size(xyzlist,1) == 3
    
    [uv,  Tuv]  = DLTnormalization(uvlist)
    [xyz, Txyz] = DLTnormalization(xyzlist)
    
    matrix = np.zeros((np.size(xyzlist,0)*3,12))
    
    for n in range(np.size(uvlist,0)):
        xyz1 = np.hstack([xyz[n], 1])
        u    = uv[n,0]
        v    = uv[n,1]
        w    = 1
        matrix[3*n,:]   = np.hstack([ 0*xyz1, w*xyz1,   -v*xyz1])
        matrix[3*n+1,:] = np.hstack([-w*xyz1, 0*xyz1,    u*xyz1])
        matrix[3*n+2,:] = np.hstack([ v*xyz1,-u*xyz1,    0*xyz1])
        
    [_,D,V] = np.linalg.svd(matrix)
    V = V[-1] #takes last vector

#    # Remember the fact that the points are in front of the camera
    if V[-1] < 0:
        V = -V
        
#    print "Minimum singular value", D[-1]

    M = np.dot(np.linalg.inv(Tuv), 
               np.dot(np.vstack([V[0:4],V[4:8],V[8:12]]), Txyz))
 
#    print "Check"
    for n in range(np.size(uvlist,0)):
        uv  = np.array([uvlist[n,0],   uvlist[n,1], 1])
        xyz = np.array([xyzlist[n,0], xyzlist[n,1], xyzlist[n,2], 1])
        ans = np.dot(M,xyz.T)
        if np.max(np.abs(uv - ans / ans[2])) > 1e-3:
            warnings.warn("Discrepancy of more than 1e-3 found in DLT", Warning)

    # This normalization is applied so that the third element of the resulting
    # UV-vector is the distance from the XYZ-point to the center of projection 
    M = M / np.linalg.norm(M[2,:3])

    # This matrix provides the derivatives of the UV-coordinates with respect
    # to the XYZ coordinates
    #                        / dU/dx \
    #                        | dU/dy |
    #                        | dU/dz |
    # dMdX * M * XYZ = w^2 * | dV/dx | 
    #                        | dV/dy |
    #                        \ dV/dz / 
    #
    # Where [u,v,w]^T = M * [x,y,z,1]^T
    # 
    dMdX = np.array([[-M[2,0],       0,    M[0,0]],
                     [-M[2,1],       0,    M[0,1]],
                     [-M[2,2],       0,    M[0,2]],
                     [      0, -M[2,0],    M[1,0]],
                     [      0, -M[2,1],    M[1,1]],
                     [      0, -M[2,2],    M[1,2]]])
#    print M, "\n", np.linalg.det(M[:,:3]), "\n"
    return (M, dMdX, np.linalg.det(M[:,:3]), D[0]/D[-2], D[-1])

def DLTnormalization(pointslist):
    """
    This normalization procedure was suggested in:: "Multiple view geometry in
    computer vision" by Hartley and Zisserman, and is needed to make the
    problem of finding the direct linear transform converge better
    
    The idea is transforming the set of points so that their average is zero
    and their distance to the origin is in average sqrt(nb. of coordinates).
    
    Parameters
    ----------
    pointslist: numpy.array
        A matrix with the size (N,C) where N is the number of points and C is 
        the number of coordinates
        
    Returns
    -------
    normalized_points: numpy.array
        A matrix with the same size as the input with the normalized coordinates
    T: numpy.array
        A matrix with the form (C+1, C+1) representing the transformation to be
        used with homogeneous coordinates
        
    Examples
    --------
    >>> pointslist = np.array([[0,0],
    ...                        [0,1],
    ...                        [1,1],
    ...                        [1,0]])
    >>> [normpoints, T] = DLTnormalization(pointslist)
    >>> (np.mean(normpoints,0) == 0).all()
    True
    
    >>> homogeneouspoints = np.ones((4,3))       #must convert to homog. coords.
    >>> homogeneouspoints[:,:-1] = normpoints
    >>> np.dot(np.linalg.inv(T),homogeneouspoints.T).T[:,:-1] #inverse transform
    array([[ 0.,  0.],
           [ 0.,  1.],
           [ 1.,  1.],
           [ 1.,  0.]])
    """
    ncoords = np.size(pointslist,1)+1
    t           = np.mean(pointslist,0)
    s           = np.mean(norm(pointslist - t)) / np.sqrt(ncoords-1) 
    T           = np.eye(ncoords) / s 
    T[-1,-1]    = 1
    T[:-1,-1]   = -t / s

    return ((pointslist - t)/s, T)
    
    
def pointInHexa(p,hexapoints):
    """
    Taking a set of points defining a hexahedron in the conventional order,
    this function tests of the point is inside this hexahedron by:
   
    For each face:
        - calculate normal pointing outwards
        - verify if point is "behind" the plane defined by the face
       
    Parameters
    ----------
    p : numpy.array (N, 3)
        List of points to be tested
    hexapoints : numpy.array (8, 3)
        List of points defining an hexahedron, must obey the conventional order
        of defining hexas
       
    Returns
    -------
    1 
        if points lies inside the hexahedron
    0 
        otherwise
        
    Examples
    --------
    >>> hexapoints = np.array([[0,0,0], 
    ...                        [0,1,0], 
    ...                        [0,1,1], 
    ...                        [0,0,1], 
    ...                        [1,0,0], 
    ...                        [1,1,0], 
    ...                        [1,1,1], 
    ...                        [1,0,1]])
    >>> p = np.array([[  0,  0,  0], 
    ...               [0.5,0.5,0.5], 
    ...               [  2,  0,  0]])
    >>> pointInHexa(p, hexapoints)
    array([ 0.,  1.,  0.])
    """

   
    if p.ndim > 1:
        truthtable = np.ones(len(p))
    else:
        truthtable = 1
              
    for n in range(6):
        vout = np.cross(hexapoints[HEXA_CONN_PARTIAL[n,0]] - 
                        hexapoints[HEXA_CONN_PARTIAL[n,2]],
                        hexapoints[HEXA_CONN_PARTIAL[n,1]] - 
                        hexapoints[HEXA_CONN_PARTIAL[n,2]])
        
        truthtable = truthtable * \
                    (np.einsum("ij,j->i",
                               p - hexapoints[HEXA_CONN_PARTIAL[n,2]], 
                               vout) < 0)

    return truthtable                      
                 
def quadInterpolation(p,pquad,values): 
    """
    Performs barycentric interpolation extended to the case of a planar quad
    
    Parameters
    ----------
    p : numpy.array (N, 3)
    pquad : numpy.array (4, 3)
    values : numpy.array (4, M)
    
    Returns
    -------
    result : numpy.array (N, M)        
    
    Examples
    --------
    >>> pquad = [[-1,-1,0],
    ...          [+1,-1,0],
    ...          [+1,+1,0],
    ...          [-1,+1,0]]
    >>> p     = [[0,-1  ,0],
    ...          [0,-0.5,0],
    ...          [0,   0,0],
    ...          [0,  +1,0]]
    >>> values = [0,0,1,1]
    >>> quadInterpolation(np.array(p), 
    ...                   np.array(pquad),
    ...                   np.array(values))
    array([ 0.  ,  0.25,  0.5 ,  1.  ])
    
    For interpolation of vectors
    
    >>> values = [[-1,-1,0],
    ...           [+1,-1,0],
    ...           [+1,+1,0],
    ...           [-1,+1,0]]
    >>> quadInterpolation(np.array(p), 
    ...                   np.array(pquad),
    ...                   np.array(values))
    array([[ 0. , -1. ,  0. ],
           [ 0. , -0.5,  0. ],
           [ 0. ,  0. ,  0. ],
           [ 0. ,  1. ,  0. ]])
    """
    if p.ndim > 1:
        npts = np.size(p,0)
    else:
        npts = 1
        
    Su = triangleArea(p,pquad[3],pquad[2])
    Sr = triangleArea(p,pquad[2],pquad[1])
    Sd = triangleArea(p,pquad[0],pquad[1])
    Sl = triangleArea(p,pquad[0],pquad[3])
    den = (Su + Sd)*(Sr + Sl)
    c1 = np.reshape(Sr*Su/den,(npts,1,1))
    c2 = np.reshape(Sl*Su/den,(npts,1,1))
    c3 = np.reshape(Sl*Sd/den,(npts,1,1))
    c4 = np.reshape(Sr*Sd/den,(npts,1,1))
    
    return (c1*values[0] + c2*values[1] + 
            c3*values[2] + c4*values[3]).squeeze()
           
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
        if t == 0:
            print "Elapsed time too short to measure"
            return 0
        if n is None:
            print "Elapsed time: %f seconds" % t
            return t
        else:
            print "Can execute: %f calculations / second" % (n/t)
            return n/t
           
def quadArea(p1,p2,p3,p4): 
    """
    Returns the area of a quadrilateral defined by its edge points. The 
    following assumptions are made:
        - All points lie in the same plane        
        - The quadrilateral is convex
        
    Parameters
    ----------
    p1, p2, p3, p4 : numpy.array (N,3)
        Points or list of points defining quadrilaterals
        
    Returns
    -------
    result : (N)
        The area of the quadrilaterals
        
    Examples
    --------
    >>> pts    = np.array([[0,0,0], 
    ...                    [0,1,0], 
    ...                    [0,1,1], 
    ...                    [0,0,1]])
    >>> quadArea(pts[0], pts[1], pts[2], pts[3])
    1.0
    """
    return triangleArea(p1,p2,p3) + triangleArea(p1,p3,p4)
       
def displayProfile(filename):
    """
    Internal utility to profile a module and display the function calls sorted 
    by the cumulative time taken by each of them. This is a tool for development 
    of the code.
    
    Parameters
    ----------
    filename : string
        Name of the module to be profiled. The module must have some executable
        part.
    """
    import pstats
    import os
    os.system("python -m cProfile -o autogenprofile.txt " + filename)
    p = pstats.Stats("autogenprofile.txt")
    p.strip_dirs().sort_stats('cumulative').print_stats(70)

if __name__ == "__main__":
    print "Will execute doctest"
    import doctest
    doctest.testmod()
    print "If nothing was printed, Utils module is running ok"