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
import copy
import Utils

# Global constants
GLOBAL_NDIM  = 3
GLOBAL_TOL   = 1e-8

class Component(object):
    """
    The class component is a representation of elements used in the simulation.
    
    Most of its methods are abstract (throw a NotImplementedError) because the
    really useful classes are its derivatives::
    
    * :class:`~.Core.Assembly`
    * :class:`~.Core.Part`
    * :class:`~.Core.Line`
    
    However, this class exists to stipulate a common interface for all elements
    in simulation, allowing a tree-like nesting of them.
    
    Some attributes (x, y, z, origin, id) are not directly changeable (for 
    obvious reasons), only with geometrical transforms, etc, so no setter is
    implemented, and if you try to change them, you will get an error.
    
    There is also a implementation of the visitor pattern using the 
    :meth:`~Core.Component.acceptVisitor` method
    """
    
    componentCounter          = 0
   
    def __init__(self):
        self._id                        = Component.componentCounter
        self.name                       = str(self._id)
        Component.componentCounter     += 1
        self._origin                    = np.array([0,0,0])
        self._x                         = np.array([1,0,0])
        self._y                         = np.array([0,1,0])
        self._z                         = np.array([0,0,1])
        self.parent                     = None
        # Phyisical properties
        self.sellmeierCoeffs            = None
        self.indexOfRefraction          = 1
        # Raytracing properties
        self.terminalOnFOVCalculation   = True
        self.terminalAlways             = False
        self.reflectAlways              = False
        self.lightSource                = False
        self.tracingRule                = None
        
    @property
    def id(self):               return self._id
    @property
    def x(self):                return self._x
    @property
    def y(self):                return self._y
    @property
    def z(self):                return self._z
    @property
    def origin(self):           return self._origin 
       
    def getIndexOfRefraction(self, wavelength):
        """
        Returns the index of refraction of the material given the wavelength
        (or a list of them)
        
        If Sellmeier coefficients are given, calculation will be performed
        based on wavelength. 
        `Reference: <http://en.wikipedia.org/wiki/Sellmeier_equation>`
        """
        if self.reflectAlways:
            return 0
        if self.terminalAlways:
            return -1
        if (self.tracingRule == RayBundle.TRACING_FOV) and \
            (self.terminalOnFOVCalculation):
            return -1
        if (self.tracingRule == RayBundle.TRACING_FOV) and \
            (self.lightSource):
            return 1
        
        if self.sellmeierCoeffs is None:
            return self.indexOfRefraction
            
        w2 = wavelength ** 2
        Nc = np.size(self.sellmeierCoeffs,0)
                       
        return np.sqrt(1 + \
                np.sum((w2 * self.sellmeierCoeffs[:,0].reshape(Nc,1,1)) / \
                       (w2 - self.sellmeierCoeffs[:,1].reshape(Nc,1,1)),0)).squeeze()
                           
    def getParentIndexOfRefraction(self, wavelength):
        """
        Shortcut for requesting the index of refraction of the parent element
        (what, in a well-constructed environment, means the ambient index of
        refraction)
        """
        if self.reflectAlways:
            return 1
        if self.terminalAlways:
            return -1
        if (self.tracingRule == RayBundle.TRACING_FOV) and \
            (self.terminalOnFOVCalculation):
            return -1
        if (self.tracingRule == RayBundle.TRACING_FOV) and \
            (self.lightSource):
            return 1
            
        return self.parent.getIndexOfRefraction(wavelength)

    def intersections(self, p0, p1, tol = GLOBAL_TOL):
        """
        This is a method used specifically for ray tracing. Its inputs are:
        p0, p1 - np.array([[x0, y0, z0],
                           [x1, y1, z1],
                            ...
                           [xn, yn, zn]])
        Defining n segments, which will be tested for intersection with the
        polygons defined in the structure.
        
        *By definition* this method will not search for intersections with lines,
        if there is any in the assembly.
        """
        return None
    
    def acceptVisitor(self, visitor):
        """
        This method is a provision for the `Visitor Pattern 
        <http://http://en.wikipedia.org/wiki/Visitor_pattern>`_ and is used
        for traversing the tree.
        
        Some possible uses are the display or the saving routine.
        
        *If you are inheriting from this class* and your node is non-terminal,
        please override this method
        """
        visitor.visit(self)
    
    def translate(self, vector):
        """
        This method should be used when there is a change in the component
        position. This method operates only with the origin position, and
        delegates the responsibility to the inheriting class by means of the
        :meth:`~Core.Component.translateImplementation()` method.
        
        Inputs::
        
        vector
            Vector to translate the component. A 3-component numpy array with
            x, y and z coordinates
        
        """
        self._origin     = self._origin + vector
        self.translateImplementation(vector)
        self.clearData()
        
    def translateImplementation(self, part2):
        """
        This method must be implemented by the interested inheriting class in
        case a translation affects its internals.
        
        For example: a class with a vector of points P will probably need to
        update that to P+part2
        
        This is a way of implementing the `Chain of Responsibility 
        <http://http://en.wikipedia.org/wiki/Chain-of-responsibility_pattern>` 
        pattern, so that these geometrical operations are executed recursively.
        
        *This is a protected method, do not use it unless you are inheriting
        from this class!*
        """
        raise NotImplementedError
        
    def rotate(self, angle, axis, pivotPoint=None):
        """
        This method should be used when there is a change in the component
        position. This method operates only with the origin and the x, y and z
        vectors. It delegates the responsibility to the inheriting class by 
        means of the :meth:`~Core.Component.rotateImplementation()` method.
        
        Inputs::
        
        angle
            Angle (in radians)
        axis
            Vector around which the rotation occurs. A 3-component numpy array 
            with x, y and z coordinates
        pivotPoint
            Point in space around which the rotation occurs. If not given, 
            rotates around origin. A 3-component numpy array with x, y and z 
            coordinates
        """
        if (np.abs(angle) < GLOBAL_TOL):
            return
        
        if pivotPoint is None:
            pivotPoint = self._origin
            
        self._origin     = Utils.rotatePoints(self.origin,angle,axis,pivotPoint)
        self._x          = Utils.rotateVector(self.x,angle,axis)
        self._y          = Utils.rotateVector(self.y,angle,axis)
        self._z          = Utils.rotateVector(self.z,angle,axis)
        self.rotateImplementation(angle,axis,pivotPoint)
        self.clearData()  
        
    def rotateImplementation(self, angle, axis, pivotPoint):
        """
        This method must be implemented by the interested inheriting class in
        case a rotation affects its internals.
        
        For example: a class with a vector of points P will probably need to
        update them accordingly using the following code::
            
            P = Utils.rotatePoints(P,angle,axis,pivotPoint)
            
        This is a way of implementing the `Chain of Responsibility 
        <http://http://en.wikipedia.org/wiki/Chain-of-responsibility_pattern>` 
        pattern, so that these geometrical operations are executed recursively.
        
        *This is a protected method, do not use it unless you are inheriting
        from this class!*
        """
        raise NotImplementedError
        
    def alignTo(self,x_new,y_new,z_new=None):
        """
        This method allows the alignment of the part to a specific direction
        (this is very useful in optical systems definition).
        
        The implementation assumes that at least two *orthogonal* vectors are
        given. There is an assertion to guarantee that.
        
        With the new vector base, a rotation matrix M is calculated, and the
        `formulation to convert rotation matrix to axis-angle 
        <http://http://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis-angle>`_
        is used for convenience (as then the method :meth:`~Core.Component.rotate`
        can be directly called).
        
        Obs: No concerns about code efficiency are made, as this method will
        probably not be used all the time.
        """
        if y_new is not None and z_new is None:
            z_new = np.cross(x_new,y_new)
        if y_new is None and z_new is not None:
            y_new = np.cross(z_new,x_new)
        
        x_new = x_new / (np.dot(x_new,x_new))**0.5
        y_new = y_new / (np.dot(y_new,y_new))**0.5
        z_new = z_new / (np.dot(z_new,z_new))**0.5
        
        # Verification that the base is orthonormal
        assert np.dot(x_new,y_new) == 0 and np.dot(y_new,z_new) == 0 and \
                np.dot(x_new,z_new) == 0
      
        Xnew = np.vstack([x_new,y_new,z_new])
        Xold = np.array([self.x,
                         self.y,
                         self.z])
        M   = np.linalg.solve(Xold,Xnew)
        assert (np.linalg.det(M) - 1)**2 < GLOBAL_TOL # property of rotation Matrix
        
        # Formulation from Wikipedia (See documentation above)
        D,V = np.linalg.eig(M)
        D = np.real(D)
        V = np.real(V)

        # Verifies that the matrix M is a rotation Matrix
        assert ((D-1)**2 < GLOBAL_TOL).any() 
        
        axis  = np.squeeze(V[:,(D-1)**2 < GLOBAL_TOL].T)
        angle = np.arccos((np.trace(M)-1)/2)

        self.rotate(angle,axis)
    
    def clearData(self):
        """
        This method must be implemented by each inheriting class. Its function 
        is to avoid classes having inconsistent data after a geometric transform.
        
        For example: a camera has a mapping funcion calculated from raytracing,
        then the user moves this camera, making the mapping invalid. 
        
        When a rotation or translation is called the clearData method is also
        called, and the class is in charge of cleaning all data that is now
        not valid anymore.
        """
        raise NotImplementedError

class Part(Component):
    """
    This implementation of the Component class is the representation of a surface
    using triangle elements. 
    
    This is supposed to be the standard element in PIVSim, as raytracing with
    such surfaces is relatively easy and plotting is also made easy by using 
    libraries such as VTK, Matplotlib and OpenGL.
    
    Another benefit is the possibility of directly reading this topology from a
    STL file, that can be exported from a CAD program.
    """
    def __init__(self):
        Component.__init__(self)
        self.name                       = 'Part ' + str(self.id)
        self.points                     = np.array([])
        self.connectivity               = np.array([])
        self.normals                    = None
        self.color                      = None
        self.opacity                    = 0.5
        self.visible                    = True
        # Variables for raytracing
        self._bounds                    = None
        self._triangleVectors           = None
        self._trianglePoints            = None
        self._triangleNormals           = None
        self._triVectorsDots            = None
        
    @property
    def bounds(self):
        if self._bounds is None:
            self._computeRaytracingData()
        return self._bounds
            
    @property
    def triangleVectors(self):
        if self._triangleVectors is None:
            self._computeRaytracingData()
        return self._triangleVectors
        
    @property
    def trianglePoints(self):
        if self._trianglePoints is None:
            self._computeRaytracingData()
        return self._trianglePoints
        
    @property
    def triangleNormals(self):
        if self._triangleNormals is None:
            self._computeRaytracingData()
        return self._triangleNormals
    
    @property
    def triVectorsDots(self):
        if self._triVectorsDots is None:
            self._computeRaytracingData()
        return self._triVectorsDots
    
    def _computeRaytracingData(self):
        """
        Computes data that will be used for raytracing, such as::
        
        _bounds
            Points that define a bounding box around the polygon.
            
            Format: [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        
        _triangleVectors
            Vectors defining the triangle sides (V1 and V2), in addition, there
            are some products defined to determine if a point in the triangle
            plane is inside the triangle or not.
            
            Format: [V1,V2, dot(V1,V1), dot(V1,V2), dot(V2,V2), (UV**2 - UU*VV)]
            
        _trianglePoints
            A shortcut, with the coordinates of each triangle, instead of the
            point-connectivity lists
            
        _triangleNormals
            Vectors that are normal to the triangle surfaces (this is needed even
            if normals were already determined)
            
            Format: [Normals(raw), Normals(normalized)
        """
        if len(self.points) > 0:
            xmin = [min(self.points[:,0]), \
                    min(self.points[:,1]), \
                    min(self.points[:,2])]
            xmax = [max(self.points[:,0]), \
                    max(self.points[:,1]), \
                    max(self.points[:,2])]
            self._bounds = np.array([xmin,xmax])
            
            Ptriangles      = self.points[self.connectivity]
            V1              = Ptriangles[:,1] - Ptriangles[:,0]
            V2              = Ptriangles[:,2] - Ptriangles[:,0]
            N               = np.cross(V1,V2)
            Nnorm           = Utils.normalize(N)
            UU              = np.sum(V1*V1,1)
            UV              = np.sum(V1*V2,1)
            VV              = np.sum(V2*V2,1)
            UVden           = (UV**2 - UU*VV)
            self._triangleVectors    = np.array([V1,V2])
            self._triVectorsDots     = np.array([UU,UV,VV,UVden])
            self._trianglePoints     = Ptriangles
            self._triangleNormals    = np.array([N,Nnorm])
        else:
            self._bounds = np.zeros((2,3))
                
    def intersections(self, p0, p1, tol = GLOBAL_TOL):
        """ Method for finding intersections between polygon and segment
        Inputs:
        p0 - segment initial point - list of 3 component list
        p1 - segment final point   - list of 3 component list
        
        The method returns only the first intersection between the line and
        the polygons. The result is given as five lists::
        
        lineParameter
            This is used to indicate how far the intersection point is from the
            segment starting point, if 0, the intersection is at p0 and if 1, the
            intersection is at p1
            
            *Iff* the parameter is > 1 (999), no intersection was found
            
        intersectionCoordinates
            This is where the intersections are found
            
        triangleIndexes
            This is the index of the triangle where the intersection was found.
            If no intersection found, will return 0, *but attention*, the only
            way to guarantee that no intersection was found is when the 
            lineParameter is zero.
        
        normals
            The normal vector at the intersection point (if the surface is
            defined with normals at vertices, interpolation is performed).
            
        part
            A list with references to this object. This is, in this case, 
            redundant, but that makes the function signature uniform with the
            `:class:~Core.Assembly`
                
        This method is intended for use in raytracing algorithms, as there is
        an initial, fast verification to see if there is a chance of any triangle
        in the polygon to be intersected, then, if it is the case, it executes 
        expensive search.
               
        Special cases when intersecting with individual triangles:
        - if line is contained on triangle plane, will ignore
        - if intersection is at p0, will not return p0
        
        Algorithm adapted from http://geomalgorithms.com/a06-_intersect-2.html
        """
        #
        # Some assertions to guarantee that the input data is correct:
        #
        assert p0.ndim == 2 # assert this is a numpy list of coordinates
        assert p1.ndim == 2 # assert this is also a numpy list
        assert np.size(p0,1) == np.size(p1,1) # assert list lengths are equal
        assert np.size(p0,0) == np.size(p1,0) # assert coordinate system is equal
        #
        # Start dumb search, step 1 - determine if line intercept triangle plane
        #
        Ptriangles                  = self.trianglePoints
        [V1,V2]                     = self.triangleVectors
        [UU,UV,VV,UVden]            = self.triVectorsDots
        [N,Nnorm]                   = self.triangleNormals
       
        # Some variable definitions to make latter code more readable:
        ntris = np.size(N,0)   # Number of triangles in surface
        nlins = np.size(p0,0)  # Number of lines in surface
        V     = p1 - p0
        
        # First we need to do a dot product between each vector and all triangle
        # normals
        V = V.reshape(nlins,1,GLOBAL_NDIM)
        den = np.sum(N*V,2)
        #
        # When we find a case of den == 0, it means that the line is parallel to
        # the triangle, a case which we'll ignore.
        #
        # Numpy seems to deal better with infinity than NaN, then:
        den[(den == 0)] = np.inf
        
        #
        # We now must find the vector that goes from a triangle point to the
        # initial point of the line
        #
        P0 = p0.reshape(nlins,1,GLOBAL_NDIM)
        V0 = Ptriangles[:,0] - P0
        num = np.sum(N*V0,2)
        #
        # Now the parameter T can be calculated. The formula is:
        #
        #  T   =   N dot V0
        #         ----------
        #          N dot V
        #
        T   =  num / den 
        T_0 = copy.deepcopy(T)
        # Now, in order to find the intersection point between the intersection
        # plane and the line, we have::
        # P = P0 + T*V
        T = T.reshape(nlins,ntris,1)
        np.tile(T.T,(1,1,GLOBAL_NDIM))
        P = P0 + T * V
        
        # Finally we have to check if the points are inside the triangles. This
        # is theoretically an easy task, as it's just projecting the vector
        # at the triangle sides:
        #
        #      P0      V2
        #        o----------->
        #         \  in     /
        #          \       /  beyond         U = P - P0
        #        V1 \     /
        #            \   /    (P)
        #             \ /
        #              part2 
        U       = P - Ptriangles[:,0]
        
        UW      = np.sum(V1 * U,2)
        VW      = np.sum(V2 * U,2)

        S1 = (UV*VW - VV*UW) / UVden
        T1 = (UV*UW - UU*VW) / UVden
        
        # Now we have to mark all points that do not lie in the triangles::
        #
        # the first line will check if the point is really in the given segment,
        # there is an important check to eliminate if the intersection is at
        # the beginning of the segment
        #
        # The second and third lines check if the point is really at the triangle
        #
        
        [Ii,Ij] = np.nonzero((T_0 <= tol)+(T_0 > 1+tol)+ \
                              (S1 < -tol)+(T1 < -tol)+ \
                              (S1 + T1 > 1 + tol))
                              
        T_0[Ii,Ij] = 999
        #
        # Finally, we take only the first intersection of the line with the
        # polygon, and arrange data for returning
        #
        triangleIndexes         = np.argmin(T_0,1)
        lineParameters          = T_0[range(nlins),triangleIndexes]
        intersectionCoords      = P[range(nlins),triangleIndexes,:]
        # Assures outputs p1 as coordinates, when intersection is not found
        intersectionCoords[lineParameters > 1]  = p1[lineParameters > 1]
        
        if self.normals is None:
            normals                 = Nnorm[triangleIndexes]
        else:
            normals                 = self._calculateNormals(triangleIndexes,
                                                             intersectionCoords)
         
        # This, when uncommented, assures that no normals are given if no
        # intersection is found
        #normals[lineParameters > 1] = 0*normals[lineParameters > 1]
                                                             
        return [lineParameters, \
                intersectionCoords,\
                triangleIndexes,\
                normals,
                np.array([self]*nlins)]\
        
    def _calculateNormals(self,triangleIndexes,intersectionCoords):
        """ 
        This method returns a 3-element list corresponding to the normalized 
        normal vector. It returns the interpolation (using barycentric coordinates) 
        of the normals on the triangle vertices - use this if representing 
        lenses, etc
        
        Inputs::
        
        triangleIndexes
            indexes of the triangles (numpy array)
        intersectionCoords
            coordinates of the intersection point (numpy array of 3 elements array)
            
        *WARNING* - will return a result even if point is not on the polygon
                      
        """
        triangleCoords  = self.points[self.connectivity[triangleIndexes]]
        normals         = self.normals[self.connectivity[triangleIndexes]]
        
        # Calculation of the barycentric coordinates for each point
        lambdas         = Utils.barycentricCoordinates(intersectionCoords,
                            triangleCoords[:,0],
                            triangleCoords[:,1],
                            triangleCoords[:,2])
        
        # Barycentric interpolation
        result = np.tile(lambdas[:,0],(3,1)).T * np.array(normals[:,0,:]) + \
                 np.tile(lambdas[:,1],(3,1)).T * np.array(normals[:,1,:]) + \
                 np.tile(lambdas[:,2],(3,1)).T * np.array(normals[:,2,:])
                 
        # As a side-effect, normals must be normalized after barycentric interp
        result = result / np.tile(np.sum(result*result,1)**0.5, (3,1)).T
 
        return result
        
    def translateImplementation(self, part2):
        """
        This method is in charge of updating the position of the point cloud
        (provided it exists) when the Part is translated.
        
        There is an exception handling because there is the possibility that 
        the part is translated before the points are defined. This is extremely
        unlikely, but should not stop the program execution.
        """
        try:
            self.points = self.points + part2
        except TypeError:
            # There is no problem if a translation is executed before the points
            # are defined
            pass
            
    def rotateImplementation(self, angle, axis, pivotPoint):
        """
        This method is in charge of updating the position of the point cloud
        (provided it exists) when the Part is rotated.
        
        There is an exception handling because there is the possibility that 
        the part is translated before the points are defined. This is extremely
        unlikely, but should not stop the program execution.
        
        In case the surface is defined with normals on vertices (thus making
        self.normals not None), these vectors are rotated.
        """
        try:
            self.points = Utils.rotatePoints(self.points,angle,axis,pivotPoint)
        except TypeError:
            # There is no problem if a rotation is executed before the points
            # are defined
            pass
            
        if self.normals is not None:
            self.normals = Utils.rotateVector(self.normals,angle,axis)
                        
    def clearData(self):
        """
        Implement this method whenever your object possesses geometrical features
        that are calculated from their interaction with the ambient (e.g. - any
        raytraced features). This method is called for all spatial transformations
        """
        self._bounds                            = None
        self._triangleVectors                   = None
        self._trianglePoints                    = None
        self._triangleNormals                   = None
  
class Line(Component):
    """
    This class is used for representation of 1D elements, i.e. lines and curves
    in the 3D space.
    
    *Warning* - do not change the self.bounds property value. This is an indication
    that this class does not take part in ray tracing activities
    """
    def __init__(self):
        Component.__init__(self)
        self.name                       = 'Line '+str(self._id)
        self.points                     = np.array([])
        self.color                      = None
        self.opacity                    = 0.5
        self.visible                    = True
        
    @property
    def bounds(self):
        """
        This signals the ray tracing implementation that no attempt should be
        made to intersect rays with lines
        """
        return None
        
    def translateImplementation(self, part2):
        """
        This method is in charge of updating the position of the point cloud
        (provided it exists) when the Line is translated.
        
        There is an exception handling because there is the possibility that 
        the line is translated before the points are defined. This is extremely
        unlikely, but should not stop the program execution.
        """
        try:
            self.points = self.points + part2
        except TypeError:
            # There is no problem if a translation is executed before the points
            # are defined
            pass
            
    def rotateImplementation(self, angle, axis, pivotPoint):
        """
        This method is in charge of updating the position of the point cloud
        (provided it exists) when the Line is rotated.
        
        There is an exception handling because there is the possibility that 
        the line is translated before the points are defined. This is extremely
        unlikely, but should not stop the program execution.
        """
        try:
            self.points = Utils.rotatePoints(self.points,angle,axis,pivotPoint)
        except TypeError:
            # There is no problem if a rotation is executed before the points
            # are defined
            pass
     
    def clearData(self):
        """
        In the pure implementation of the line, there are no features to be 
        deleted when a geometrical operation is executed
        """
        pass
  
class Assembly(Component):
    """
    The assembly class is a non-terminal node in the Components tree. It carries
    almost no properties of its own, but makes sure that all geometrical 
    transformations are propagated to its subcomponents.
    
    The subcomponents are stored in a numpy array. This is definetely not
    always needed (except when fancy slicing is desired), but this keeps
    the consistency all along the code, making all lists instances of
    numpy.ndarray.
    """
    def __init__(self):
        self._items                     = np.array([])
        self._bounds                    = None
        self._tracingRule               = None
        Component.__init__(self)
        self.name                       = 'Assembly '+str(self._id)
    
    @property
    def tracingRule(self):
        return self._tracingRule
        
    @tracingRule.setter
    def tracingRule(self, rule):
        self._tracingRule = rule
        for part in self._items:
            part.tracingRule = rule
      
    @property
    def bounds(self):
        if self._bounds is None:
            [mini,maxi] = np.zeros((2,3))
             
            if len(self._items) > 0:
                for part in self.items:
                    [xmin,xmax] = part.bounds
                    mini = mini * (mini < xmin) + xmin * (mini > xmin)
                    maxi = maxi * (maxi > xmax) + xmax * (maxi < xmax)
            self._bounds = np.array([mini,maxi])  
        return self._bounds
    
    @property
    def items(self):
        return self._items
    
    @items.setter
    def items(self,value):
        self._items = value
        for part in self._items:
            part.parent = self
            
    @items.deleter
    def items(self):
        del self._items
        
    def insert(self, component, n = None):
        """
        Adds element at the component list. If no n parameter is given, the
        element is added at the end of the list, otherwise it is added at the
        n-th position.
        
        The method returns the list length.
        """
        if n is None:
            self._items = np.append(self._items, component)
        else:
            self._items = np.insert(self._items, n, component)
            
        component.parent = self
        return len(self._items)
        
    def remove(self, n):
        """
        Remove the element at the n-th position of the component list. This
        also de-registers this assembly as its parent.
        """
        self._items[n].parent = None
        self._items = np.delete(self._items, n)
        
    def acceptVisitor(self, visitor):
        """
        This method is a provision for the `Visitor Pattern 
        <http://http://en.wikipedia.org/wiki/Visitor_pattern>`_ and is be used
        for traversing the tree.
        
        As an assembly has sub-elements, we must iterate through them
        """
        visitor.visit(self)
        for part in self._items:
            part.acceptVisitor(visitor)
      
    def _boundingBoxTest(self, bounds, p0, p1):
        """ Determines if the line segment intersects the box bounding the polygon
        Inputs:
        bounds
            a 2x3 vector with:: [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        p0
            segment initial point - 3 element list
        p1
            segment final point   - 3 element list
        
        Returns:
        +----------------------+----+
        | if intersection   |  1 |
        +----------------------+----+
        |if not intersection - 0
        
        The algorithm implemented here was taken from:
        author = {Amy Williams and Steve Barrus and R. Keith and Morley Peter 
        Shirley},
        title = {An efficient and robust ray-box intersection algorithm},
        journal = {Journal of Graphics Tools},
        year = {2003},
        volume = {10},
        pages = {54}
        
        It was written in a way to accept a list of lines.
        
        In the code, there is the following trick::
        
            V[V == 0] = GLOBAL_TOL
        
        Which seems to work to avoid the creation of NaNs in the calculation.
        """
        [xmin,xmax] =  bounds

        V = p1 - p0

        V[V == 0] = GLOBAL_TOL

        T1 = (xmin - p0) / V
        T2 = (xmax - p0) / V

        Tmin = T1 * (V >= 0) + T2 * (V < 0)
        Tmax = T2 * (V >= 0) + T1 * (V < 0)

        eliminated1 = (Tmin[:,0] > Tmax[:,1]) + (Tmin[:,1] > Tmax[:,0])

        Tmin[:,0] = Tmin[:,0] * (Tmin[:,1] <= Tmin[:,0]) + \
                    Tmin[:,1] * (Tmin[:,1] > Tmin[:,0])
                    
        Tmax[:,0] = Tmax[:,0] * (Tmax[:,1] >= Tmax[:,0]) + \
                    Tmax[:,1] * (Tmax[:,1] < Tmax[:,0])
                    
                
        eliminated2 = (Tmin[:,0] > Tmax[:,2]) + (Tmin[:,2] > Tmax[:,0])

        Tmin[:,0] = Tmin[:,0] * (Tmin[:,2] <= Tmin[:,0]) + \
                    Tmin[:,2] * (Tmin[:,2] > Tmin[:,0])
                    
        Tmax[:,0] = Tmax[:,0] * (Tmax[:,2] >= Tmax[:,0]) + \
                    Tmax[:,2] * (Tmax[:,2] < Tmax[:,0])
                    
        eliminated3 = (Tmin[:,0] > 1) + (Tmax[:,0] < 0)

        return 1 - (eliminated1 + eliminated2 + eliminated3)
          
    def intersections(self, p0, p1, tol = GLOBAL_TOL):
        """
        This is a method used specifically for ray tracing. Its inputs are:
        p0, p1 - np.array([[x0, y0, z0],
                           [x1, y1, z1],
                            ...
                           [xn, yn, zn]])
        Defining n segments, which will be tested for intersection with the
        polygons defined in the structure.
        
        *By definition* this method will not search for intersections with lines,
        if there is any in the assembly.
        """
        nlins               = np.size(p0,0)
        ndim                = np.size(p0,1)
        
        lineParameter       = np.zeros(nlins) + 999
        coordinates         = copy.deepcopy(p1)
        triangleNumber      = np.zeros(nlins)
        normalVector        = np.zeros((nlins,ndim))
        intersectedSurface  = np.array([None] * nlins)
        
        for part in self._items:    
            if part.bounds is not None:        
                boxTest = self._boundingBoxTest(part.bounds, p0, p1)
                if np.sum(boxTest) > 0:
                    p0_refined = p0[boxTest == 1]
                    p1_refined = p1[boxTest == 1]
                    
                    [lineT, coords, triInd, N, partList]  = \
                                part.intersections(p0_refined, p1_refined, tol)
                    
                    # Create a mask of elements which must be substituted
                    lineParameter_temp                  = np.zeros(nlins) + 999
                    lineParameter_temp[boxTest == 1]    = lineT
                    # This mask is used for the arrays that contain the answer
                    maskLong         = lineParameter > lineParameter_temp
                    # This mask is for the result of part.intersections
                    maskShort        = lineParameter[boxTest == 1] > lineT
                    lineParameter[maskLong]         = lineT[maskShort]
                    coordinates[maskLong]           = coords[maskShort]
                    triangleNumber[maskLong]        = triInd[maskShort]
                    normalVector[maskLong]          = N[maskShort]
                    intersectedSurface[maskLong]    = partList[maskShort]
                else:
                    pass # if nothing was found, nothing was found
              
        return [lineParameter, 
                coordinates, 
                triangleNumber, 
                normalVector, 
                intersectedSurface]
      
    def translateImplementation(self, vector):
        """
        This method iterates the translation to all the items found in the list,
        making the `:meth:~Core.Component.translate` method be executed through
        the whole components tree.
        """
        for part in self._items:
            part.translate(vector)
               
    def rotateImplementation(self, angle, axis, pivotPoint):
        """
        This method iterates the rotation to all the items found in the list,
        making the `:meth:~Core.Component.rotate` method be executed through
        the whole components tree.
        """
        for part in self._items:
            part.rotate(angle, axis, pivotPoint)
    
    def clearData(self):
        """
        This method iterates the clearData to all the items found in the list,
        making the `:meth:~Core.Component.clearData` method be executed through
        the whole components tree.
        """
        self._bounds = None
        for part in self._items:
            part.clearData()

class RayBundle(Assembly):
    """
    This class represents an arbitrary number of light rays. The reason for
    having them wrapped in a single class is that it allows full vectorization
    of the raytracing procedure.
    
    This can be called a discardable class
    """
    TRACING_FOV               = 1
    TRACING_LASER_REFLECTION  = 0
    
    def __init__(self):
        Assembly.__init__(self)
        self.name                       = 'Bundle ' + str(self._id)
        # Ray tracing configuration
        self.maximumRayTrace            = 10
        self.stepRayTrace               = 0.1
        self.preAllocatedSteps          = 3
        self.wavelength                 = None
        self.startingPoints             = None
        self.initialVectors             = None
        # Ray tracing statistics
        self.rayPaths                   = None
        self.rayIntersections           = None
        self.rayLength                  = None
        self.steps                      = None
        self.finalIntersections         = None
        
        
    @property
    def bounds(self):
        """
        This ensures that no attempt is made to find intersections between this
        element and the light rays in a raytracing procedure
        """
        return None
        
    def insert(self, initialVector, initialPosition = None, wavelength = 532e-9):
        """
        This method is used to insert new rays in the bundle. This can work in
        several ways::
        
        1) A single ray is inserted::
        
        >>> bundle = RayBundle()
        >>> bundle.insert(np.array([1,0,0]), np.array([0,0,0]))
        
        2) Several rays are inserted, each with a starting point::
        
        >>> bundle.insert(np.array([[1,0,0],[1,0,0]]), \\
                          np.array([[0,0,0],[0,0,1]]))
        
        3) Several rays are inserted, with a common starting point::
        
        >>> bundle.insert(np.array([[1,0,0],[0,1,0]], \\
                          np.array([0,0,0]))
                          
        Notes: 
        
        *   If the starting point is omitted, it will assume that the rays
            departs from the origin of the bundle
            
        *   This method was not made to be efficient, so it should be used 
            ideally only once, and not in a loop
            
        *   This method destroys data that was already ray-traced, if any
        """
        # clear data, as it would be really difficult to manage rays with 
        # different number of points
        self.clearData()
        
        # Determine how many vectors were received
        if initialVector.ndim == 1:
            nrays = 1
            assert len(initialVector) == GLOBAL_NDIM
        else:
            nrays = np.size(initialVector,0)
        
        # Adjust initialPosition according to the given conditions
        if initialPosition is None:
            initPos = np.tile(self.origin, (nrays,1))
        else:
            if initialPosition.ndim == 1 and nrays > 1:
                initPos = np.tile(initialPosition, (nrays,1))
            else:
                initPos = initialPosition
                
        # properly add the newly received vectors to the stack
        if self.startingPoints is not None:
            self.startingPoints = np.vstack(self.startingPoints, initPos)
            self.initialVectors = np.vstack(self.initialVectors, initialVector)
        else:
            self.startingPoints = initPos
            self.initialVectors = initialVector
            
        # create the new storage for ray paths
        self.rayPaths = copy.deepcopy(self.startingPoints)
        # This looks not careful, but in reality enforces that wavelength
        # is either scalar or the same length as the startingPoints
        self.wavelength = np.ones(np.size(self.startingPoints,0)) * wavelength
        
    def delete(self, n):
        """
        This method is not implemented, as deleting a single ray requires 
        many matrix reshapings. 
        
        If you want to delete rays, use the clear method.
        """
        return NotImplementedError
    
    def clear(self):
        self.wavelength                 = None
        self.startingPoints             = None
        self.initialVectors             = None
        self.clearData()

    def translateImplementation(self, part2):
        """
        This method changes the rays starting points, and waits for clear data
        to delete all ray tracing related information.
        """
        try:
            self.startingPoints             = self.startingPoints + part2
        except TypeError:
            pass # there might be no starting points registered
               
    def rotateImplementation(self, angle, axis, pivotPoint):
        """
        This method does nothing, because a rotation or a translation of light 
        rays may completely change the light paths, requiring a full new ray
        tracing
        """
        try:
            self.startingPoints = Utils.rotatePoints(self.startingPoints, 
                                                     angle, axis, pivotPoint)
            self.initialVectors = Utils.rotateVector(self.initialVectors,
                                                      angle, axis)
        except TypeError:
            pass
        
    def trace(self, tracingRule = TRACING_FOV):
        # Make sure everything is clear
        self.clearData()
        
        # Routine to find the top element in the hierarchy
        topComponent = self
        while topComponent.parent is not None:
            topComponent = topComponent.parent
        # Tells all components which kind of ray tracing this is
        topComponent.tracingRule = tracingRule
        
        # Shortcut to the number of rays being traced:
        nrays = np.size(self.startingPoints, 0)
        
        # Initialize variables
        distance                    = np.zeros(nrays)
        currVector                  = copy.deepcopy(self.initialVectors)
        self.finalIntersections     = np.empty(nrays,dtype='object')
        # Do matrix pre-allocation to store ray paths
        rayPoints   = np.empty((self.preAllocatedSteps, nrays, GLOBAL_NDIM),
                               dtype='double')
        rayIntersc  = np.empty((self.preAllocatedSteps, nrays, 1),
                               dtype='object')
        step        = 0
        rayPoints[step  ,:,:] = copy.deepcopy(self.startingPoints)
        rayPoints[step+1,:,:] = self.startingPoints + \
                                self.stepRayTrace*self.initialVectors
        stepsize = np.ones(nrays) * self.stepRayTrace

        while np.sum(stepsize) > GLOBAL_TOL:
            # Increase matrix size (if this is done too often, performance is
            # really, really bad. So adjust self.preAllocatedSteps wisely
            if step + 2 >= np.size(rayPoints,0):
                # Reallocate points vector
                temp = np.empty((step + self.preAllocatedSteps, 
                                 nrays, GLOBAL_NDIM), dtype = "double")
                temp[range(np.size(rayPoints,0))] = rayPoints
                rayPoints = temp
                # Reallocate intersections vector
                temp = np.empty((step + self.preAllocatedSteps, 
                                 nrays, GLOBAL_NDIM), dtype = "object")
                temp[range(np.size(rayIntersc,0))] = rayIntersc
                rayIntersc = temp
              
            # Ask for the top assembly to intersect the rays with the whole
            # Component tree, will receive results only for the first 
            # intersection of each ray
            [t, \
            coords, \
            _, \
            N, \
            surfaceRef] = topComponent.intersections(rayPoints[step], \
                                                     rayPoints[step+1], \
                                                     GLOBAL_TOL)
            self.finalIntersections[t <= 1] = \
                                            surfaceRef[t <= 1]
            rayIntersc[step+1,:,:] = np.reshape(surfaceRef,(nrays,1))
        
            # Calculate the distance ran by the rays
            distance = distance + \
                       t * (t <= 1) * stepsize + \
                       (t > 1) * stepsize
        
            # Calculate the next vectors
            currVector  = self.calculateNextVectors(currVector, 
                                                    t, 
                                                    N, 
                                                    surfaceRef)
            
            # Calculate next step size
            stepsize = \
                (self.stepRayTrace * \
                    (distance + self.stepRayTrace <= self.maximumRayTrace) + \
                    (self.maximumRayTrace - distance) * \
                    (distance + self.stepRayTrace > self.maximumRayTrace)) * \
                     Utils.norm(currVector)
                
            # Calculate next inputs:
            step              = step + 1
            rayPoints[step]   = coords
            rayPoints[step+1] = rayPoints[step] + \
                                currVector * np.tile(stepsize,(GLOBAL_NDIM,1)).T
                   
        # Now, clean up the mess with the preallocated matrix:
        self.rayPaths                   = rayPoints[range(step+1)]
        self.rayIntersections           = rayIntersc[range(step+1)]
        # Save ray tracing statistics
        self.rayLength                  = distance
        self.steps                      = step
        self.finalIntersections         = surfaceRef
        
        # And create lines to represent the rays
        self._items = np.empty(nrays,"object")
        for n in range(nrays):
            self._items[n] = Line()
            self._items[n].parent = self
            self._items[n].points = self.rayPaths[:,n,:]
            self._items[n].color  = Utils.metersToRGB(self.wavelength[n])
            
    def calculateNextVectors(self, currVector, t, N, surface):
        """
        TODO
        """
        if (t > 1 + GLOBAL_TOL).all():
            return currVector
        
        # Keep these assertions here if you are unsure that you are getting
        # the correct input data
        assert(Utils.aeq(N, Utils.normalize(N)))
        assert(Utils.aeq(currVector, Utils.normalize(currVector)))
        
        #=========================================
        # Calculate as if all rays were reflected
        #=========================================
        reflected = currVector - 2 * N * np.tile(np.sum(N * currVector,1),(3,1)).T
        
        #=========================================
        # Calculate refractions
        #=========================================
        # Important calculation:
        NdotV       = np.sum(currVector * N, 1)
        # Properties:
        # NdotV < 0 => Ray entering the surface (normals point outwards)
        # NdotV > 0 => Ray exiting the surface
        # NdotV = 0 => Should not happen, as the intersection algorithm
        #               rejects that. Spurious cases will be filtered later
        
        Nsurf     = np.zeros_like(surface)
        Nparent   = np.zeros_like(surface)
        N1        = np.ones_like(surface)
        N2        = np.ones_like(surface)
        cosTheta1 = -NdotV
        cosTheta2 = np.inf * N1
        refracted = np.zeros_like(currVector)
        
        if (t <= 1 + GLOBAL_TOL).any():
            # Get the indexes of refraction, result is filtered to reduce load
            # on the numpy.vectorize method
            for n, surf in enumerate(surface):
                if surf is not None:
                    Nsurf[n]   = surf.getIndexOfRefraction(self.wavelength[n])
                    Nparent[n] = surf.getParentIndexOfRefraction(self.wavelength[n])
                                                                
            # If entering surface, N1 is the external index of refraction, N2 is
            # the internal
            N1[(NdotV < 0) * (t <= 1) * (Nsurf > 0) * (Nparent > 0)] = \
                Nparent[(NdotV < 0) * (t <= 1) * (Nsurf > 0) * (Nparent > 0)]
            N2[(NdotV < 0) * (t <= 1) * (Nsurf > 0) * (Nparent > 0)] = \
                Nsurf[(NdotV < 0) * (t <= 1) * (Nsurf > 0) * (Nparent > 0)]
            # and vice versa
            N1[(NdotV > 0) * (t <= 1) * (Nsurf > 0) * (Nparent > 0)] = \
                Nsurf[(NdotV > 0) * (t <= 1) * (Nsurf > 0) * (Nparent > 0)]
            N2[(NdotV > 0) * (t <= 1) * (Nsurf > 0) * (Nparent > 0)] = \
                Nparent[(NdotV > 0) * (t <= 1) * (Nsurf > 0) * (Nparent > 0)]
                
            
            # formulation taken from: http://en.wikipedia.org/wiki/Snell's_law
            cosTheta2 = 1 - (1 - cosTheta1**2) * ((N1 / N2) ** 2)
            cosTheta2[cosTheta2 >= 0] = (cosTheta2[cosTheta2 >= 0])**0.5
            
            refracted = np.tile(N1/N2,(GLOBAL_NDIM,1)).T * currVector + \
                        np.tile(np.sign(cosTheta1) * \
                                ((N1/N2)*np.abs(cosTheta1) - cosTheta2), \
                                (GLOBAL_NDIM,1)).T * \
                                N
            refracted = Utils.normalize(refracted)
        
        #=========================================
        #   Big if block to sort the cases out
        #=========================================
        # First, assume rays were undisturbed:
        result = currVector
        # Then substitute those who were successfully refracted / pass through
        result[(cosTheta2 <= 1 + GLOBAL_TOL) * (Nsurf > 0) * (Nparent > 0)] = \
            Utils.normalize(refracted[(cosTheta2 <= 1 + GLOBAL_TOL) * \
                                      (Nsurf > 0) * (Nparent > 0)])
        # Then zero those rays who found a terminal
        result[(Nsurf == -1) * (Nparent == -1)] = \
            0 * result[(Nsurf < 0) * (Nparent < 0)]
        # Then put reflected rays
        result[Nsurf == 0] = reflected[Nsurf == 0]
        result[cosTheta2 < 0] = reflected[cosTheta2 < 0]
        
        return result
        
    def clearData(self):
        """
        This method removes all elements from the self.items list and the ray
        pathes
        """
        self.items                      = np.array([])
        self.rayPaths                   = copy.deepcopy(self.startingPoints)
        self.rayLength                  = None
        self.steps                      = None
    
class Plane(Part):
    """
    This is a convenience class that inherits from Part and represents
    a rectangle. There are also convenience methods to  make coordinate
    transformation.
    
    As a default, the plane is defined as::
    
    y (first coordinate)
    ^  
    |--------------+
    |              |h 
    |              |e
    |              |i
    |              |g
    |              |t
    |              |h
    +--------------+--> z (second coordinate)
         length
         
    To navigate in the plane, one can use the following coordinate system::
    [y',z'], where 0 <= y' <=1 and 0 <= z' <=1
    
    As a default, the X vector is the normal for the triangles.
    
    This class can represent *only* rectangles, so that most of its methods
    are greatly simplified. If you need to represent a parallelogram, you
    would have to implement your own class.
    """
    PARAMETRIC_COORDS = np.array([[+0,-0.5,-0.5],
                                  [+0,+0.5,-0.5],
                                  [+0,+0.5,+0.5],
                                  [+0,-0.5,+0.5]])
    def __init__(self, length = 1, heigth = 1):
        Part.__init__(self)
        self.name           = 'Plane '+str(self._id)
        self._length        = length
        self._heigth        = heigth
        self.connectivity   = np.array([[0,1,2], [0,2,3]])
        self.normals        = None
        self._dimension     = None
        self._resize()
    
    @property
    def length(self):
        return self._length
    @length.setter
    def length(self, HEXA_CONN_PARTIAL):
        self._length = HEXA_CONN_PARTIAL
        self._resize()
        
    @property
    def heigth(self):
        return self._heigth
    @heigth.setter
    def heigth(self, h):
        self._heigth = h
        self._resize()
        
    @property
    def dimension(self):
        return self._dimension
    @dimension.setter
    def dimension(self, d):
        self._length        = d[0]
        self._heigth        = d[1]
        self._dimension     = d
        self._resize()        
    
    def _resize(self):
        """
        Convenience function to position the points of the plane and set
        up internal variables
        """
        self._dimension = np.array([self._heigth,  self._length])
        self.points = Plane.PARAMETRIC_COORDS[:,1:3] * np.tile(self.dimension,(4,1))
        self.points = np.tile(self.points[:,0],(GLOBAL_NDIM,1)).T * self.y + \
                      np.tile(self.points[:,1],(GLOBAL_NDIM,1)).T * self.z 
          
    def parametricToPhysical(self,coordinates):
        """
        Transforms a 2-component vector in the range 0..1 in sensor coordinates
        Normalized [y,z] -> [x,y,z] (global reference frame)
        
        Vectorization for this method is implemented.
        """
        # Compensate for the fact that the sensor origin is at its center
        coordinates = self.dimension*(coordinates - 0.5)
#        print "Coordinates"
#        print coordinates
        
        if coordinates.ndim == 1:
#            print "ndim == 1"
            return self.origin + coordinates[0] * self.y + \
                                 coordinates[1] * self.z
        else:
#            print "ndim > 1"
#            print np.tile(coordinates[:,0],(GLOBAL_NDIM,1)).T
#            print np.tile(coordinates[:,1],(GLOBAL_NDIM,1)).T
            return self.origin + \
                   np.tile(coordinates[:,0],(GLOBAL_NDIM,1)).T * self.y + \
                   np.tile(coordinates[:,1],(GLOBAL_NDIM,1)).T * self.z
                    
    def physicalToParametric(self,coords):
        """
        Transforms a 3-component coordinates vector to a 2-component vector
        which value falls in the range 0..1 in sensor coordinates
        Normalized [y,z] -> [x,y,z] (global reference frame)
        
        Vectorization for this method is implemented.
        """
        part2 = coords - self.origin
        if coords.ndim == 1:
            py = (np.dot(self.y,part2) / self.dimension[0]) + 0.5
            pz = (np.dot(self.z,part2) / self.dimension[1]) + 0.5
            return np.array([py,pz])        
        else:
            nvecs = np.size(part2,0)
            py = (np.sum(np.tile(self.y,(nvecs,1).T*part2,1) / \
                          self.dimension[0])) + 0.5
            pz = (np.sum(np.tile(self.z,(nvecs,1).T*part2,1) / \
                          self.dimension[0])) + 0.5
            return np.array([py,pz]).T
    
class Volume(Part):
    """
    This class is used to represent a general hexahedron. Even though some
    methods will force it to become a cuboid (where all angles are right), such
    as::
    
    * :meth:`~Core.Volume.width`
    * :meth:`~Core.Volume.depth`
    * :meth:`~Core.Volume.heigth`
    * :meth:`~Core.Volume.dimension`
    
    These methods can be safely ignored a set of points can be directly given,
    allowing quadrilaterally-faced hexas to be represented (check 
    `Wikipedia's article <http://en.wikipedia.org/wiki/Hexahedron>` for more
    information)
    """
    PARAMETRIC_COORDS = np.array([[+0,+0.5,+0.5],
                                   [+0,-0.5,+0.5],
                                   [+0,-0.5,-0.5],
                                   [+0,+0.5,-0.5],
                                   [+1,+0.5,+0.5],
                                   [+1,-0.5,+0.5],
                                   [+1,-0.5,-0.5],
                                   [+1,+0.5,-0.5],])
    def __init__(self, heigth = 1, depth = 1, width = 1):
        """
        This is another convenience class to represent volumes (a hexahedron).
        
        The following conventions are used::
        
        x
            Dimension of heigth
        y
            Dimension of depth
        z
            Dimension of width
            
        The hexahedron is built the following way:
                            X
                            ^
                     +------|----------+
                     |      |          |
                     |      |          | heigth
                     |      |          |
                   h /------|----------/
                  t /       |         /
                 p /        +--------/---------> Z
                e /        /        /
               d /--------/--------/
                        width   
                        /
                       /
                      part2  Y
                      
        The point numbering convention is::
        
           6--------5
          /        /|    (X)
         /        / |    ^
        7--------4  |    |
        |        |  |    +-> (Z)
        |   2    |  1   /
        |        | /   part2 
        |        |/    (Y)
        3--------0
        """
        Part.__init__(self)
        self.name           = 'Volume '+str(self._id)
        self._width         = width
        self._depth         = depth
        self._heigth        = heigth
        self._dimension     = [heigth, depth, width]
        # normals pointing outside 
        self.connectivity   = np.array([[1,4,0],[1,5,4], # normal +z
                                        [7,2,3],[7,6,2], # normal -z
                                        [0,7,3],[0,4,7], # normal +y
                                        [6,5,1],[6,1,2], # normal -y
                                        [4,6,7],[4,5,6], # normal +x
                                        [0,3,2],[2,1,0]]) # normal -x
        self.normals        = None
        self._resize()

    @property
    def width(self):
        return self._width
    @width.setter
    def width(self, w):
        self._width = w
        self._resize()
        
    @property
    def depth(self):
        return self._depth
    @depth.setter
    def depth(self, d):
        self._length = d
        self._resize()
        
    @property
    def heigth(self):
        return self._heigth
    @heigth.setter
    def heigth(self, h):
        self._heigth = h
        self._resize()
        
    @property
    def dimension(self):
        return self._dimension
    @dimension.setter
    def dimension(self, d):
        self._heigth        = d[0]
        self._depth         = d[1]
        self._width         = d[2]
        self._dimension     = d
        self._resize()        
          
    def _resize(self):
        self.points = Volume.PARAMETRIC_COORDS * np.tile(self.dimension,(8,1))
        self.points = (np.reshape(self.points[:,0],(8,1,1)) * self.x).squeeze() + \
                      (np.reshape(self.points[:,1],(8,1,1)) * self.y).squeeze() + \
                      (np.reshape(self.points[:,2],(8,1,1)) * self.z).squeeze() 
        
    def physicalToParametric(self, c):
        """
        Transforms a vector or a list of vectors in parametric coordinates with
        the following properties:
        
        [x,y,z] (global) -> [x',y',z'] (parametrical)
        
        0 <= x',y',z' <= 1 if [x,y,z] lies inside the volume 
        """                
        return Utils.hexaInterpolation(c, \
                                       self.points, \
                                       Volume.PARAMETRIC_COORDS[:,1:3])
    
    def parametricToPhysical(self, p):
        """
        Transforms a vector or a list of vectors in parametric coordinates with
        the following properties:
        
        [x,y,z] (global) -> [x',y',z'] (parametrical)
        
        0 <= x',y',z' <= 1 if [x,y,z] lies inside the volume 
        """     
        return Utils.hexaInterpolation(p, \
                                       Volume.PARAMETRIC_COORDS[:,1:3], \
                                       self.points)
          
    def pointInHexa(self,p):
        """
        This is intended as a lightweigth test for checking if a point (or
        a set of them) lies inside an hexahedron. This uses the algorithm
        implemented in :mod:`Utils`.
        """
        return Utils.pointInHexa(p,self.points)
  
if __name__=="__main__":
    """
    Code for unit testing basic functionality of classes in the Core module
    """
    print ""
    print "*********************************************************************"
    print "*************       PIVSim Core module unit test      ***************"
    print "*********************************************************************"
    #=====================================
    # Simplified geometry creation - cube
    #=====================================
    points = [[0,0,0],
              [1,0,0],
              [1,1,0],
              [0,1,0],
              [0,0,1],
              [1,0,1],
              [1,1,1],
              [0,1,1]]
    
    # normals pointing outside
    conn = [[5,7,4],[5,6,7], # normal +z
           [3,2,1],[0,3,1], # normal -z
           [3,6,2],[6,3,7], # normal +y
           [1,5,4],[4,0,1], # normal -y
           [5,1,6],[1,2,6], # normal +x
           [7,0,4],[7,3,0]] # normal -x
    #        
    # Create values for putting normals at edges
    #
    edgenormals = []
    for n in range(8):
        norm = points[n] - np.array([0.5,0.5,0.5])
        norm = norm/np.sqrt(np.dot(norm,norm))
        edgenormals.append(norm)
    edgenormals = np.array(edgenormals)
    
    print "*******  Creation of the part and an assembly containing it  ********"
    part                = Part()
    part.points         = np.array(points)
    part.connectivity   = np.array(conn)
    assembly            = Assembly()
    assembly.insert(part)
    print "Successfully created a project tree"
    print part.name
    print assembly.name
    #
    # Test refraction coefficient
    #
    print "************     Index of refraction calculation   ******************"
    part.sellmeierCoeffs      = np.array([[1.03961212, 6.00069867e-15],
                                          [0.23179234, 2.00179144e-14],
                                          [1.01046945, 1.03560653e-10]])
    assert Utils.aeq(part.getIndexOfRefraction(532e-9), 1.51947, 1e-3)
    assert Utils.aeq(part.getIndexOfRefraction(486e-9), 1.52238, 1e-3)
    assert Utils.aeq(part.getIndexOfRefraction(np.ones(10)*532e-9),
                                               np.ones(10)*1.51947, 1e-3)
    part.sellmeierCoeffs      = None
    assert Utils.aeq(part.getIndexOfRefraction(532e-9), 1)
    print "************              Intersection test        ******************"
    #         hit            miss             hit                      miss
    
    p0 = [[0.5,0.5,-1],  [1.5,1.5,-1],  [0.999999,2 ,0.999999],  [-1,-1e-6,-1e-6]]
    p1 = [[0.5,0.5, 2],  [1.5,1.5, 2],  [1       ,-1,1],         [ 2,    0,    0]]
    t0 = [      0,             10,              0,                       10]
    p0 = np.array(p0)
    p1 = np.array(p1)
    t0 = np.array(t0)
    # Parameter for the speed test
    repetitions = 10
    cases       = np.size(p0,0)
    triangles   = np.size(part.points,0)
    tic         = Utils.Tictoc()
    # Verify that the intersections were correctly found
    tic.tic()
    [t, coords, inds, norms, refs] = part.intersections(p0, p1, GLOBAL_TOL)
    tic.toc(cases*triangles)
    assert sum((t > t0))
    
    # Assert that the assembly will give exactly the same answer as the part
    tic.tic()
    [t2, coords2, inds2, norms2, refs2] = assembly.intersections(p0, p1, GLOBAL_TOL)
    tic.toc(cases*triangles)
    assert Utils.aeq(t,t2)
    assert Utils.aeq(coords,coords2)
    assert Utils.aeq(inds,inds2)
    
    print "*************    Intersection performance test     ******************"
    p0 = np.tile(p0, (repetitions,1))
    p1 = np.tile(p1, (repetitions,1))
    t0 = np.tile(t0, repetitions)
    
    print "Intersections with line using the method from Core.Part"
    _ = part.bounds # This forces the pre-calcs for raytracing (optional)
    tic.tic()
    [t, coords, inds, norms, refs] = part.intersections(p0, p1, GLOBAL_TOL)
    tic.toc(cases*triangles*repetitions)
    assert sum((t > t0))
    
    print "Intersections with line using the method from Core.Assembly"
    tic.tic()
    [t2, coords2, inds2, norms2, refs2] = assembly.intersections(p0, p1, GLOBAL_TOL)
    tic.toc(cases*triangles*repetitions)
    assert Utils.aeq(t,t2)
    assert Utils.aeq(coords,coords2)
    assert Utils.aeq(inds,inds2)
    
    print "Intersections with line using the method after random rotation"
    angle  = np.random.rand()
    axis   = np.array([1,1,1]) / np.linalg.norm(np.array([1,1,1]))
    pivot  = assembly.origin
    assembly.rotate(angle,axis,pivot)
    p0 = Utils.rotatePoints(p0,angle,axis,pivot)
    p1 = Utils.rotatePoints(p1,angle,axis,pivot)
    _ = part.bounds
    tic.tic()
    [t, coords, inds, norms, refs] = assembly.intersections(p0, p1, GLOBAL_TOL)
    tic.toc(cases*triangles*repetitions)
    assert Utils.aeq(t, t2)
    print "Reference result from PIVSim part2.0 - 71500 polygon intersection / second"

    #=============================
    # Testing the provided normals
    #=============================
    print "************    Normal vector calculation test     ******************"
    assembly.rotate(-angle, axis,pivot)
    p0 = [[0.5,0.5,-1.],  [0.5,0.5,0.5],  [-1,0.5,0.5],  [.5,0.5,0.5]]
    p1 = [[0.5,0.5,0.5],  [0.5,0.5,1.5],  [.5,0.5,0.5],  [2.,0.5,0.5]]
    t0 =    [[0,0,-1],       [0,0,1],       [-1,0,0],       [1,0,0]]
    p0 = np.array(p0)
    p1 = np.array(p1)
    t0 = np.array(t0)
    [t, coords, inds, norms, refs] = assembly.intersections(p0, p1, GLOBAL_TOL)
    assert Utils.aeq(norms, t0)
    
    #
    # Test of interpolated normals (result must be the same)
    #
    part.normals = edgenormals
    [t, coords, inds, norms, refs] = assembly.intersections(p0, p1, GLOBAL_TOL)
    assert Utils.aeq(norms,t0)
    
    #================================
    # Testing of the ray bundle class
    #================================
    print "************         Basic ray tracing test        ******************"
    part.terminalOnFOVCalculation   = False
    part.terminalAlways             = False
    part.reflectAlways              = False
    part.lightSource                = False
    print "Items:"
    print assembly.items
    assembly.tracingRule            = RayBundle.TRACING_LASER_REFLECTION
    
    bundle = RayBundle()
    bundle.translate(np.array([0.5,0.5,0.5]))
    bundle.insert(np.array([[1,0,0],[0,1,0],[0,0,1]]))
    assembly.insert(bundle)
    
    print "Tracing ray bundle"
    print "Pre allocated steps : ", bundle.preAllocatedSteps
    print "Step ray trace      : ", bundle.stepRayTrace
    tic.tic()
    bundle.trace()
    tic.toc()
    print "Ray lengths         : ", bundle.rayLength
    print "Number of steps     : ", bundle.steps
    print bundle.rayPaths[-1]
    
    bundle.preAllocatedSteps          = 10
    bundle.stepRayTrace               = 5
    print "Tracing ray bundle"
    print "Pre allocated steps : ", bundle.preAllocatedSteps
    print "Step ray trace      : ", bundle.stepRayTrace
    
    print bundle.startingPoints
    print bundle.initialVectors
    bundle.rotate(np.pi/4, np.array([0,0,1]))
    print bundle.initialVectors
    
    tic.tic()
    bundle.trace()
    tic.toc()
    print "Ray lengths         : ", bundle.rayLength
    print "Number of steps     : ", bundle.steps
    print bundle.rayPaths[-1]
    
    print "************  Testing geometrical operations       ******************"
    theorypoints = np.array([[0,-1,-1],
                             [0,+1,-1],
                             [0,+1,+1],
                             [0,-1,+1]])*0.5
    m = Plane()
    assert Utils.aeq(m.origin, np.zeros(3))
    assert Utils.aeq(m.x, np.eye(3)[0])
    assert Utils.aeq(m.y, np.eye(3)[1])
    assert Utils.aeq(m.z, np.eye(3)[2])
    coords = m.parametricToPhysical(np.array([[0.5,0.5],[1,1],[0,0]]))
    assert Utils.aeq(coords[0], np.zeros(3))
    assert Utils.aeq(coords[1], m.parametricToPhysical(np.array([1,1])))
    assert Utils.aeq(coords[2], m.parametricToPhysical(np.array([0,0])))
    assert Utils.aeq(m.points, theorypoints)
    
    print "testing translation"
    m.translate(np.array([0,1,1])*0.5)
    
    theorypoints = np.array([[0,-0,-0],
                             [0,+2,-0],
                             [0,+2,+2],
                             [0,-0,+2]])*0.5
    assert Utils.aeq(m.points, theorypoints)
    
    print "testing rotation aroung axis passing through origin"
    m.translate(np.array([0,-1,-1])*0.5)
    m.rotate(np.pi/2,np.array([1,0,0]))

    theorypoints = np.array([[0,+1,-1],
                             [0,+1,+1],
                             [0,-1,+1],
                             [0,-1,-1]])*0.5                          
    assert Utils.aeq(m.points, theorypoints)    

    m.rotate(2*np.pi,np.array([1,0,0]))
    assert Utils.aeq(m.points, theorypoints)    
    
    m.rotate(-np.pi/2,np.array([1,0,0]))
    theorypoints = np.array([[0,-1,-1],
                             [0,+1,-1],
                             [0,+1,+1],
                             [0,-1,+1]])*0.5
                             
    assert Utils.aeq(m.points, theorypoints) 

    m.rotate(np.pi/2,np.array([0,1,0]))
    theorypoints = np.array([[-1,-1,-0],
                             [-1,+1,-0],
                             [+1,+1,+0],
                             [+1,-1,+0]])*0.5
    assert Utils.aeq(m.points, theorypoints)  
    assert Utils.aeq(m.x, np.array([0,0,-1]))  
    m.rotate(-np.pi/2,np.array([0,1,0]))
    
    print "testing the align to axis function"
    m.alignTo(np.array([1,0,0]),np.array([0,1,0]))
    theorypoints = np.array([[0,-1,-1],
                             [0,+1,-1],
                             [0,+1,+1],
                             [0,-1,+1]])*0.5
    assert Utils.aeq(m.points, theorypoints)   
       
    m.alignTo(np.array([-1,0,0]),np.array([0,1,0]))
    theorypoints = np.array([[0,-1,+1],
                             [0,+1,+1],
                             [0,+1,-1],
                             [0,-1,-1]])*0.5
    assert Utils.aeq(m.points, theorypoints)   
    
    m.alignTo(np.array([1,0,0]),None,np.array([0,0,1]))
    theorypoints = np.array([[0,-1,-1],
                             [0,+1,-1],
                             [0,+1,+1],
                             [0,-1,+1]])*0.5
    assert Utils.aeq(m.points, theorypoints)                                

    print "testing rotation by axis off-origin"
    m.rotate(np.pi,np.array([1,0,0]),np.array([0,1,1])*0.5)
    theorypoints = np.array([[0,+1.5,+1.5],
                             [0,+0.5,+1.5],
                             [0,+0.5,+0.5],
                             [0,+1.5,+0.5]])
    assert Utils.aeq(m.points, theorypoints)     
    
    print "************               END OF TESTS            ******************"