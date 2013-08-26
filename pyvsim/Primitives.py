"""
.. module :: Primitives
    :platform: Unix, Windows
    :synopsis: Classes representing geometric entities or basic building blocks
    
The classes contained in this module represent the basics of geometric modelling
in pyvsim. It also contains the ray tracing engine and its models (reflection,
refraction, etc)
    
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
from __future__ import division
import numpy as np
import copy
import Utils
import Library
import Core
import weakref

# Global constants
GLOBAL_NDIM  = 3
GLOBAL_TOL   = 1e-8

class Component(Core.PyvsimObject):
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
    MIRROR                    = np.array([True,  False, False, False])
    TRANSPARENT               = np.array([False,  True, False, False])
    OPAQUE                    = np.array([False, False,  True, False])
    DUMP                      = np.array([False, False, False,  True])
    PLOTDIMS                  = -1
    
    def __init__(self):
        Core.PyvsimObject.__init__(self)
        self._origin                    = np.array([0,0,0])
        self._x                         = np.array([1,0,0])
        self._y                         = np.array([0,1,0])
        self._z                         = np.array([0,0,1])
        self.parent                     = None
        self._depth                     = None          
               
    @property
    def x(self):                return self._x
    @property
    def y(self):                return self._y
    @property
    def z(self):                return self._z
    @property
    def origin(self):           return self._origin 
    @property
    def bounds(self):           return None
    @property
    def depth(self):
        """
        Return the depth of the component within a tree
        """
        if self.parent is None:
            self._depth = 0
        else:
            self._depth = 1 + self.parent.depth
        return self._depth
                                 
    def intersections(self, p0, p1, tol = GLOBAL_TOL):
        """
        This is a method used specifically for ray tracing. The method returns 
        data about the first intersection between line segments and the 
        polygons defined in the Component. The implementation of the
        intersection is given by the inheriting classes.
        
        Parameters
        ----------
        p0, p1 - numpy.array (N x 3)
            Coordinates defining N segments by 2 points (each p0, p1 pair), 
            which will be tested for intersection with the polygons defined in 
            the structure.
        tol - double
            Tolerance used in the criteria for intersection (see documentation
            of each implementation)
            
        Returns
        -------
        None
            If no intersections are found. 
            
        Otherwise returns a list with::
            
        lineParameter
            This is used to indicate how far the intersection point is from the
            segment starting point, if 0, the intersection is at p0 and if 1, 
            the intersection is at p1
            
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
        """
        return None
        
    def translate(self, vector):
        """
        This method should be used when there is a change in the component
        position. This method operates only with the origin position, and
        delegates the responsibility to the inheriting class by means of the
        :meth:`~Core.Component.translateImplementation()` method.
        
        Parameters
        ----------
        vector : numpy.array (1 x 3)
            Vector to translate the component. An array with x, y and z 
            coordinates
        """
        self._origin     = self._origin + vector
        self.translateImplementation(vector)
        self.clearData()
        
    def translateImplementation(self, vector):
        """
        This method must be implemented by the interested inheriting class in
        case a translation affects its internals.
        
        For example: a class with a vector of points P will probably need to
        update that to P+vector
        
        This is a way of implementing the `Chain of Responsibility 
        <http://http://en.wikipedia.org/wiki/Chain-of-responsibility_pattern>` 
        pattern, so that these geometrical operations are executed recursively.
        
        *This is a protected method, do not use it unless you are inheriting
        from this class!*
        
        Parameters
        ----------
        vector : numpy.array (1 x 3)
            Vector to translate the component. An array with x, y and z 
            coordinates
                
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError
        
    def rotate(self, angle, axis, pivotPoint=None):
        """
        This method should be used when there is a change in the component
        position. This method operates only with the origin and the x, y and z
        vectors. It delegates the responsibility to the inheriting class by 
        means of the :meth:`~Core.Component.rotateImplementation()` method.
        
        Parameters
        ----------
        angle
            Angle : scalar (in radians)
        axis : numpy.array (1 x 3)
            Vector around which the rotation occurs.
        pivotPoint : numpy.array (1 x 3)
            Point in space around which the rotation occurs. If not given, 
            rotates around origin.
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
        
        Parameters
        ----------
        angle
            Angle : scalar (in radians)
        axis : numpy.array (1 x 3)
            Vector around which the rotation occurs.
        pivotPoint : numpy.array (1 x 3)
            Point in space around which the rotation occurs. If not given, 
            rotates around origin.
        
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError
        
    def alignTo(self,x_new,y_new,z_new=None, pivotPoint = None, tol=1e-8):
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
        
        Parameters
        ----------
        x_new, [y_new, z_new] : numpy.array(1,3)
            New vectors defining the orientation of the part. The vectors need
            NOT to be normalized, but MUST be orthogonal.
            
            Vectors y_new or z_new can be omitted (one at a time) and will be
            implicitly calculated
        pivotPoint : numpy.array(1,3)
            Point in space around which the rotation occurs. If not given, 
            rotates around local origin.
        tol : 1e-8
            Many checks are executed in order to guarantee that the new x, y and
            z vectors are orthogonal and normalized. This is the tolerance of
            these checks.
        
        Raises
        ------
        AssertionError
            If the vectors are not perpendicular to the given tolerance (check
            is done with a dot product), or if the rotation is performed from
            a right-handed coordinate system to a left-handed and vice-versa.
        LinAlgError
            If the calculation has other mathematical problems.
            
        """        
        if pivotPoint is None:
            pivotPoint = self.origin
        
        if y_new is not None and z_new is None:
            z_new = np.cross(x_new,y_new)
        if y_new is None and z_new is not None:
            y_new = np.cross(z_new,x_new)
        
        Xnew = Utils.normalize(np.vstack([x_new,
                                          y_new,
                                          z_new]))
        
        # Verification that the base is orthonormal
        assert (Utils.aeq(np.dot(Xnew[0],Xnew[1]),0, tol) and 
                Utils.aeq(np.dot(Xnew[0],Xnew[2]),0, tol) and
                Utils.aeq(np.dot(Xnew[2],Xnew[1]),0, tol))
        # Verify it is right-handed
        assert (Utils.aeq(np.cross(Xnew[0],Xnew[1]), Xnew[2], tol) and 
                Utils.aeq(np.cross(Xnew[0],Xnew[2]),-Xnew[1], tol) and
                Utils.aeq(np.cross(Xnew[2],Xnew[1]),-Xnew[0], tol))
      
        Xold = np.array([self.x,
                         self.y,
                         self.z])
        M   = np.linalg.solve(Xold,Xnew)
        if Utils.aeq(M, np.eye(3)):
            return

        assert (np.linalg.det(M) - 1)**2 < GLOBAL_TOL # prop of rotation Matrix
        
        # Formulation from Wikipedia (See documentation above)
        D,V = np.linalg.eig(M)
        D = np.real(D)
        V = np.real(V)

        # Verifies that the matrix M is a rotation Matrix
        assert ((D-1)**2 < GLOBAL_TOL).any() 

        axis  = np.squeeze(V[:,(D-1)**2 < GLOBAL_TOL].T)
        cosAngle = (np.trace(M)-1)/2
        # Sometimes small numeric errors are found, so must correct them,
        # otherwise np.arrccos returns nan
        if (abs(cosAngle) > 1): 
            cosAngle = np.sign(cosAngle)
        angle = np.arccos(cosAngle)

        # We can't know the right rotation, so we must check
        if Utils.aeq(Utils.rotateVector(Xold, angle, axis), Xnew):
            self.rotate(angle,axis,pivotPoint)
        else:
            self.rotate(-angle,axis,pivotPoint)
    
    def clearData(self):
        """
        This method must be implemented by each inheriting class. Its function 
        is to avoid classes having inconsistent data after a geometric transform.
        
        For example: a camera has a mapnp.ping funcion calculated from raytracing,
        then the user moves this camera, making the mapnp.ping invalid. 
        
        When a rotation or translation is called the clearData method is also
        called, and the class is in charge of cleaning all data that is now
        not valid anymore.
        """
        raise NotImplementedError

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
    PLOTDIMS                  = -1
    
    def __init__(self):
        self._items                     = np.array([], dtype = object)
        self._bounds                    = None
        self.surfaceProperty            = Component.TRANSPARENT
        Component.__init__(self)
        self.name                       = 'Assembly '+str(self._id)
        # Ray tracing properties
        self.material                   = Library.IdealMaterial(1)
        self.surfaceProperty            = Component.TRANSPARENT     
        
    def __repr__(self):
        """
        Returns a pretty-printable tree representation of the assembly
        generated recursively
        """
        string = Component.__repr__(self) + "\n|"
        if self._items is not None:
            for item in self._items:
                string = string +  (item.depth*3) * " "  + "+->"
                string = string +  item.__repr__() + "\n|"
        return string             
    
    def __iadd__(self,other):
        """
        Overloads the "+=" operator to act as an append element, or a list 
        extension
        """
        if not issubclass(type(other), Component):
            raise TypeError("Operations are only allowed between \
                             pyvsim components")       
        self.append(other)       
        return self
    
    def __isub__(self,other):
        """
        Overloads the "-=" operator to act as an remove element, or a list 
        extension
        """
        if not issubclass(type(other), Component):
            raise TypeError("Operations are only allowed between \
                             pyvsim components")
        self.remove(other)        
        return self       
        
    def __eq__(self, other):
        """
        This overloading is used in ray tracing, because it might be that the
        tracing target is given as an assembly, but the intersection always
        happens with a subcomponent of the assembly
        """
        answer = np.zeros_like(other)
        
        answer += (other is self)

        for item in self._items:
            answer += (item == other)
            
        return answer
    
    def __neq__(self, other):
        """
        This overloading is given to maintain coherence with __eq__
        """
        return 1 - (self == other) 
        
    def __getitem__(self, k):
        """
        This overloading is provided so that the assembly can be referenced by
        index, as in an array
        """
        if type(k) is str:
            for item in self._items:
                if item.name == k:
                    return item
            raise KeyError("Element ", k, "is not available")
        else:
            return self._items[k]
    
    def __setitem__(self, k, value):
        """
        This overloading is provided so that the assembly can be referenced by
        index, as in an array
        """        
        self.append(value, k)      
        
    def __delitem__(self,k):
        """
        Removes the element of a list
        
        Method provided to align assembly behavior to that of a list.
        """        
        self.remove(k)
        
    def __len__(self):
        """
        Returns the length of the items list.
        
        Method provided to align assembly behavior to that of a list.
        """
        return len(self._items)
    
    def __contains__(self, other):
        """
        Check if items list contains element
        
        Method provided to align assembly behavior to that of a list.
        """        
        return other in self._items
        
    def refractiveIndex(self, wavelength = 532e-9):
        """
        Returns the index of refraction of the material given the wavelength
        (or a list of them)
        
        Parameters
        ----------
        wavelength : scalar or numpy.array
            The wavelength of the incoming light given in *meters*
        
        Returns
        -------
        refractiveIndex : same dimension as wavelength
            The index of refraction
        """       
        return self.material.refractiveIndex(wavelength)
          
    @property
    def bounds(self):
        """
        Returns the boundaries of the assembly by finding the minimum bounding
        box aligned to the axis that contains it.
        
        The algorithm works by finding the maximum and minimum values of
        x, y and z by traversing the subcomponents.
        
        Returns
        -------
        bounds : numpy.array(2,3)
            An array containing the following elements: 
            [[xmin, ymin, zmin],
             [xmax, ymax, zmax]]
        """ 
        if self._bounds is None:
            mini =  np.ones((len(self.items),3))*1000
            maxi = -np.ones((len(self.items),3))*1000
            if len(self._items) > 0:
                for n in range(len(self._items)):
                    b = self._items[n].bounds
                    if b is not None:
                        [mini[n],maxi[n]] = b 
                self._bounds = np.array([np.min(mini,0),np.max(maxi,0)])  
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
        
    def append(self, component, n = None):
        """
        Adds element at the component list. 
        
        Parameters
        ----------
        component : Component
            The Component to be added
        n : int = None
            [Optional] The position of the component to be added. If no 
            parameter is given, the element is added at the end of the list, 
            otherwise it is added at the n-th position.
        
        Returns
        -------
        n
            The list length.
        """
        if n is None:
            n = len(self._items)
            self._items    = np.append(self._items,
                                       n, 
                                       None)             

        self._items[n] = component
            
        component.parent = weakref.proxy(self)
        self._bounds = None
        return len(self._items)
        
    def remove(self, element):
        """
        Remove the element at the n-th position of the component list. This
        also de-registers this assembly as its parent.
        
        Parameters
        ----------
        element : string, int or object
            The name, the position in the list or the object to be removed
            
        Returns
        -------
        element
            A reference to the element, if one is to re-use that.
        """
        index = None
        if type(element) is str:
            for i, elem in enumerate(self._items):
                if elem.name == element:
                    index = i
        elif issubclass(type(element), Core.PyvsimObject):
            for i, elem in enumerate(self._items):
                if elem is element:
                    index = i
        elif type(element) is int:
            index = element
        else:
            raise TypeError("Input must be either string, int or pyvsimobject")           
        
        if index is None:
            raise IndexError("index out of bounds")
        
        element             = self._items[index]
        element.parent      = None
        self._items         = np.delete(self._items, index)
        return element
        
    def acceptVisitor(self, visitor):
        """
        This method is a provision for the `Visitor Pattern 
        <http://http://en.wikipedia.org/wiki/Visitor_pattern>`  and is be used
        for traversing the tree.
        
        As an assembly has sub-elements, we must iterate through them
        """
        visitor.visit(self)
        for part in self._items:
            part.acceptVisitor(visitor)
      
    def _boundingBoxTest(self, bounds, p0, p1):
        """ 
        Determines if lines defined by the segments p0-p1 intersects the box 
        bounding the polygon
        
        The algorithm implemented was taken from:
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
        
        Parameters
        ----------
        bounds : numpy.array([[xmin,ymin,zmin],[xmax,ymax,zmax]])
            the dimensions of the bounding box
        p0 : numpy.array (N x 3)
            segment initial point - accepts simultaneous calculation of N 
            points
        p1 : numpy.array (N x 3)
            segment final point   - accepts simultaneous calculation of N 
            points
        
        Returns
        -------
        +----------------------+----+
        | if intersection      |  1 |
        +----------------------+----+
        |if not intersection   |  0 | 
        +----------------------+----+
        if N lives were given, will return a N-long numpy.array
        """
        [xmin,xmax] =  bounds

        V = p1 - p0

        V[V == 0] = GLOBAL_TOL

        T1 = (xmin - p0) / V
        T2 = (xmax - p0) / V

        Tmin = T1 * (V >= 0) + T2 * (V < 0)
        Tmax = T2 * (V >= 0) + T1 * (V < 0)

        eliminated1 = (Tmin[:,0] > Tmax[:,1]) + (Tmin[:,1] > Tmax[:,0])

        Tmin[:,0] = (Tmin[:,0] * (Tmin[:,1] <= Tmin[:,0]) + 
                     Tmin[:,1] * (Tmin[:,1] > Tmin[:,0]))
                    
        Tmax[:,0] = (Tmax[:,0] * (Tmax[:,1] >= Tmax[:,0]) + 
                     Tmax[:,1] * (Tmax[:,1] < Tmax[:,0]))
                    
                
        eliminated2 = (Tmin[:,0] > Tmax[:,2]) + (Tmin[:,2] > Tmax[:,0])

        Tmin[:,0] = (Tmin[:,0] * (Tmin[:,2] <= Tmin[:,0]) + 
                     Tmin[:,2] * (Tmin[:,2] > Tmin[:,0]))
                    
        Tmax[:,0] = (Tmax[:,0] * (Tmax[:,2] >= Tmax[:,0]) + 
                     Tmax[:,2] * (Tmax[:,2] < Tmax[:,0]))
                    
        eliminated3 = (Tmin[:,0] > 1) + (Tmax[:,0] < 0)

        return 1 - (eliminated1 + eliminated2 + eliminated3)
          
    def intersections(self, p0, p1, tol = GLOBAL_TOL):
        """
        This method searches for intersections between a given set of line
        segments and the Parts included in this Assembly. Please check the
        documentation at `:class:~Core.Part` for a better description of its
        internals.
        
        Parameters
        ----------
        p0, p1 - numpy.array (N x 3)
            Coordinates defining N segments by 2 points (each p0, p1 pair), 
            which will be tested for intersection with the polygons defined in 
            the structure.
        tol - double
            Tolerance used in the criteria for intersection (see documentation
            of each implementation)
            
        Returns
        -------
        None
            If no intersections are found. Otherwise returns a list with::
        lineParameter
            This is used to indicate how far the intersection point is from the
            segment starting point, if 0, the intersection is at p0 and if 1, 
            the intersection is at p1
            
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
                    
                    [lineT, 
                     coords,                
                     triInd, 
                     N, 
                     partList]  = part.intersections(p0_refined, 
                                                     p1_refined, 
                                                     tol)
                    
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

class Points(Component):
    """
    This class is used for representation of 0D elements, i.e. points
    in the 3D space.
    
    *Warning* - do not change the self.bounds 
    This is an indication
    that this class does not take part in ray tracing activities
    """
    PLOTDIMS                  = 0
    
    def __init__(self):
        Component.__init__(self)
        self.name                       = 'Line '+str(self._id)
        self.points                     = np.array([])
        self.connectivity               = None
        self.color                      = None
        self.opacity                    = 0.5
        self.visible                    = True
                
    def translateImplementation(self, vector):
        """
        This method is in charge of updating the position of the point cloud
        (provided it exists) when the Line is translated.
        
        There is an exception handling because there is the possibility that 
        the line is translated before the points are defined. This is extremely
        unlikely, but should not stop the program execution.
        """
        try:
            self.points = self.points + vector
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


class Line(Component):
    """
    This class is used for representation of 1D elements, i.e. lines and curves
    in the 3D space.
    
    *Warning* - do not change the self.bounds property value. This is an indication
    that this class does not take part in ray tracing activities
    """
    PLOTDIMS                  = 1
    def __init__(self):
        Component.__init__(self)
        self.name                       = 'Line '+str(self._id)
        self.points                     = np.array([])
        self.color                      = None
        self.width                      = None
        self.opacity                    = 0.5
        self.visible                    = True
                
    def translateImplementation(self, vector):
        """
        This method is in charge of updating the position of the point cloud
        (provided it exists) when the Line is translated.
        
        There is an exception handling because there is the possibility that 
        the line is translated before the points are defined. This is extremely
        unlikely, but should not stop the program execution.
        """
        try:
            self.points = self.points + vector
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

class Part(Component):
    """
    This implementation of the Component class is the representation of a surface
    using triangle elements. 
    
    This is supposed to be the standard element in np.piVSim, as raytracing with
    such surfaces is relatively easy and plotting is also made easy by using 
    libraries such as VTK, Matplotlib and OpenGL.
    
    Another benefit is the possibility of directly reading this topology from a
    STL file, that can be exported from a CAD program.
    """
    PLOTDIMS                  = 3
    
    def __init__(self):
        Component.__init__(self)
        self.name                       = 'Part ' + str(self.id)
        self.points                     = np.array([])
        self.connectivity               = np.array([])
        self.data                       = None
        self.normals                    = None
        self.color                      = None
        self.opacity                    = 0.5
        self.visible                    = True
        # Ray tracing properties
        self.material                   = Library.IdealMaterial(1)
        self.surfaceProperty            = Component.OPAQUE
        # Variables for raytracing
        self.surfaceProperty            = Component.OPAQUE
        self._bounds                    = None
        self._triangleVectors           = None
        self._trianglePoints            = None
        self._triangleNormals           = None
        self._triVectorsDots            = None
        """
        This avoids the saving of the specific ray tracing variables, which
        can consume a lot of storage while being more or less easy to be
        calculated when ray tracing is performed
        """
        self.transientFields.extend(["_bounds",
                                     "_trianglePoints"
                                     "_triangleVectors",
                                     "_triangleNormals",
                                     "_triVectorsDots"])
        
    def refractiveIndex(self, wavelength = 532e-9):
        """
        Returns the index of refraction of the material given the wavelength
        (or a list of them)
        
        Parameters
        ----------
        wavelength : scalar or numpy.array
            The wavelength of the incoming light given in *meters*
        
        Returns
        -------
        refractiveIndex : same dimension as wavelength
            The index of refraction
        """       
        return self.material.refractiveIndex(wavelength)
        
    @property
    def bounds(self):
        """
        Returns the coordinates of the aligned-to-axis-bounding box
        
        Returns
        -------
        bounds : numpy.array
            An array with the following data [xmin, xmax, ymin, ymax, zmin,
            zmax]. Defining a box aligned to axis bounding the Part
        """
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
            xmin = np.min(self.points,0)
            xmax = np.max(self.points,0)
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
        """ 
        This is a method used specifically for ray tracing. The method returns 
        data about the first intersection between line segments and the 
        polygons defined in the Component. The implementation of the
        intersection is given by the inheriting classes.
        
        This method is intended for use in raytracing algorithms, as there is
        an initial, fast verification to see if there is a chance of any 
        triangle in the polygon to be intersected, then, if it is the case, 
        it executes expensive search.
               
        Special cases when intersecting with individual triangles:
        
        - if line is contained on triangle plane, will ignore
        - if intersection is at p0, will not return p0
        
        Algorithm adapted from http://geomalgorithms.com/a06-_intersect-2.html
        
        Parameters
        ----------
        p0, p1 - numpy.array (N x 3)
            Coordinates defining N segments by 2 points (each p0, p1 pair), 
            which will be tested for intersection with the polygons defined in 
            the structure.
        tol - double
            Tolerance used in the criteria for intersection (see documentation
            of each implementation)
            
        Returns
        -------
        None
            If no intersections are found. Otherwise returns a list with::
        lineParameter
            This is used to indicate how far the intersection point is from the
            segment starting point, if 0, the intersection is at p0 and if 1, 
            the intersection is at p1
            
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
        # Retrieve data about the triangles (either pre-existing or will be 
        # calculated once)
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
        #              v 
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
        
        [Ii,Ij] = np.nonzero((T_0 <= tol)+(T_0 > 1+tol)+ 
                              (S1 < -tol)+(T1 < -tol)+ 
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
                                                             
        return [lineParameters,
                intersectionCoords,
                triangleIndexes,
                normals,
                np.array([self]*nlins)]
        
    def _calculateNormals(self,triangleIndexes,intersectionCoords):
        """ 
        This method returns a 3-element list corresponding to the normalized 
        normal vector. It returns the interpolation (using barycentric 
        coordinates) of the normals on the triangle vertices - use this if 
        representing lenses, etc
        
        *WARNING* - will return a result even if point is not on the polygon
        
        Parameters
        ----------
        triangleIndexes : numpy.array (N x 3)
            indexes of the triangles vertices (the order is important, otherwise
            the normals can be inverted. As this method is vectorized, it is
            possible to execute N calculations at the same time.
        intersectionCoords : numpy.array (N x 3)
            coordinates of the intersection points
            
        Returns
        -------
        result : numpy.array (N x 3)
            normal vectors
        """
        triangleCoords  = self.points[self.connectivity[triangleIndexes]]
        normals         = self.normals[self.connectivity[triangleIndexes]]
        
        # Calculation of the barycentric coordinates for each point
        lambdas         = Utils.barycentricCoordinates(intersectionCoords,
                            triangleCoords[:,0],
                            triangleCoords[:,1],
                            triangleCoords[:,2])
        
        # Barycentric interpolation
        result = (np.tile(lambdas[:,0],(3,1)).T * np.array(normals[:,0,:]) + 
                  np.tile(lambdas[:,1],(3,1)).T * np.array(normals[:,1,:]) + 
                  np.tile(lambdas[:,2],(3,1)).T * np.array(normals[:,2,:]))
                 
        # As a side-effect, normals must be normalized after barycentric interp
        result = Utils.normalize(result)
 
        return result
        
    def translateImplementation(self, vector):
        """
        This method is in charge of updating the position of the point cloud
        (provided it exists) when the Part is translated.
        
        There is an exception handling because there is the possibility that 
        the part is translated before the points are defined. This is extremely
        unlikely, but should not stop the program execution.
        
        Parameters
        ----------
        vector : numpy.array (1 x 3)
            Vector to translate the component. An array with x, y and z 
            coordinates
        """
        try:
            self.points = self.points + vector
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
        
        Parameters
        ----------
        angle
            Angle : scalar (in radians)
        axis : numpy.array (1 x 3)
            Vector around which the rotation occurs.
        pivotPoint : numpy.array (1 x 3)
            Point in space around which the rotation occurs. If not given, 
            rotates around origin.
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
        Implement this method whenever your object possesses geometrical 
        features that are calculated from their interaction with the ambient 
        (e.g. - any raytraced features). This method is called after all spatial 
        transformations
        """
        self._bounds                            = None
        self._triangleVectors                   = None
        self._trianglePoints                    = None
        self._triangleNormals                   = None
    
class Plane(Part):
    """
    This is a convenience class that inherits from Part and represents
    a rectangle. There are also convenience methods to  make coordinate
    transformation.
             
    To navigate in the plane, one can use the following coordinate system:
    :math:`\\left[u,v\\right]` , where 
    :math:`-1 <= u,v <=1`
    
    As a default, the :math:`\\vec{x}` vector is the normal for the triangles.
    
    This class can represent **only** rectangles, so that most of its methods
    are greatly simplified. If you need to represent a parallelogram, you
    would have to implement your own class.
    """
    PARAMETRIC_COORDS = np.array([[+0,-0.5,-0.5],
                                  [+0,+0.5,-0.5],
                                  [+0,+0.5,+0.5],
                                  [+0,-0.5,+0.5]])
    PLOTDIMS                  = 3
    def __init__(self, dimension = np.array([0,1,1]), fastInit=False):
        Part.__init__(self)
        self.name           = 'Plane '+str(self._id)
        self.connectivity   = np.array([[0,1,2], [0,2,3]])
        self.normals        = None
        self._dimension     = dimension
        if not fastInit:
            self._resize()
  
    @property
    def dimension(self):
        return self._dimension
    @dimension.setter
    def dimension(self, d):
        self._dimension     = d
        self._resize()        
    
    def _resize(self):
        """
        Convenience function to position the points of the plane and set
        up internal variables
        """
        self.points = np.einsum('ij,j->ij',
                                Plane.PARAMETRIC_COORDS, self.dimension)
        self.points = (np.tile(self.points[:,1],(GLOBAL_NDIM,1)).T * self.y + 
                       np.tile(self.points[:,2],(GLOBAL_NDIM,1)).T * self.z)
          
    def parametricToPhysical(self,coordinates):
        """
        Transforms a 2-component vector in the range -1..1 in sensor coordinates
        :math:`[u,v] \\rarrow [x,y,z]` (global reference frame)
        
        Vectorization for this method is implemented.
        """
        # This is unfortunate, because:
        # u = Zcamera
        # v = Ycamera
        # but the dimension vector is [x,y,z]
        coordinates = self.dimension[::-1][:2]*coordinates/2
#        print "Coordinates"
#        print coordinates
        
        if coordinates.ndim == 1:
#            print "ndim == 1"
            return self.origin + (coordinates[0] * self.z + 
                                  coordinates[1] * self.y)
        else:
#            print "ndim > 1"
#            print np.tile(coordinates[:,0],(GLOBAL_NDIM,1)).T
#            print np.tile(coordinates[:,1],(GLOBAL_NDIM,1)).T
            return (self.origin + 
                    np.tile(coordinates[:,0],(GLOBAL_NDIM,1)).T * self.z + 
                    np.tile(coordinates[:,1],(GLOBAL_NDIM,1)).T * self.y)
                    
    def physicalToParametric(self,coords):
        """
        Transforms a 3-component coordinates vector to a 2-component vector
        which value falls in the range -1..1 in sensor coordinates
        :math:`[x,y,z] \\rarrow [u,v]`
        
        Vectorization for this method is implemented.
        """
        vector = coords - self.origin
        if coords.ndim == 1:
            pv = 2*(np.dot(self.y,vector) / self.dimension[1]) 
            pu = 2*(np.dot(self.z,vector) / self.dimension[2])
            return np.array([pu,pv])        
        else:
            nvecs = np.size(vector,0)
            pu = (np.sum(np.tile(self.z,(nvecs,1).T*vector,1) / 
                          self.dimension[1]))
            pv = (np.sum(np.tile(self.y,(nvecs,1).T*vector,1) / 
                          self.dimension[2]))
            return 2*np.array([pu,pv]).T
    
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
    `Wikipedia's article <http://en.wikipedia.org/wiki/Hexahedron>`_ for more
    information)
    """
    PLOTDIMS          = 3
    PARAMETRIC_COORDS = np.array([[+0,+0.5,+0.5],
                                  [+0,-0.5,+0.5],
                                  [+0,-0.5,-0.5],
                                  [+0,+0.5,-0.5],
                                  [+1,+0.5,+0.5],
                                  [+1,-0.5,+0.5],
                                  [+1,-0.5,-0.5],
                                  [+1,+0.5,-0.5],])
    def __init__(self, dimension = np.array([1,1,1]), fastInit=False):
        """
        This is another convenience class to represent volumes (a hexahedron).
                      
        The point numbering convention is::
        
           6--------5
          /        /|    (X)
         /        / |    ^
        7--------4  |    |
        |        |  |    +-> (Z)
        |   2    |  1   /
        |        | /   v
        |        |/    (Y)
        3--------0
        """
        Part.__init__(self)
        self.name           = 'Volume '+str(self._id)
        self._dimension     = dimension
        # normals pointing outside 
        self.connectivity   = np.array([[1,4,0],[1,5,4], # normal +z
                                        [7,2,3],[7,6,2], # normal -z
                                        [0,7,3],[0,4,7], # normal +y
                                        [6,5,1],[6,1,2], # normal -y
                                        [4,6,7],[4,5,6], # normal +x
                                        [0,3,2],[2,1,0]]) # normal -x
        self.normals        = None
        if not fastInit:
            self._resize()
       
    @property
    def dimension(self):
        return self._dimension
    @dimension.setter
    def dimension(self, d):
        self._dimension     = d
        self._resize()        
          
    def _resize(self):
        self.points = Volume.PARAMETRIC_COORDS * np.tile(self.dimension,(8,1))
        self.points =((np.reshape(self.points[:,0],(8,1,1)) * self.x).squeeze()+ 
                      (np.reshape(self.points[:,1],(8,1,1)) * self.y).squeeze()+ 
                      (np.reshape(self.points[:,2],(8,1,1)) * self.z).squeeze()) 
        
    def expand(self, factor):
        """
        Inflates the volume by "factor"
        """
        center = np.mean(self.points,0)
        for n in range(np.size(self.points,0)):
            self.points[n] = self.points[n] + factor*(self.points[n] - center)
        
    def physicalToParametric(self, c):
        """
        Transforms a vector or a list of vectors in parametric coordinates with
        the following properties:
        
        [x,y,z] (global) -> [x',y',z'] (parametrical)
        
        0 <= x',y',z' <= 1 if [x,y,z] lies inside the volume 
        """                
        return Utils.hexaInterpolation(c, 
                                       self.points, 
                                       Volume.PARAMETRIC_COORDS[:,1:3])
    
    def parametricToPhysical(self, p):
        """
        Transforms a vector or a list of vectors in parametric coordinates with
        the following properties:
        
        [x,y,z] (global) -> [x',y',z'] (parametrical)
        
        0 <= x',y',z' <= 1 if [x,y,z] lies inside the volume 
        """     
        return Utils.hexaInterpolation(p, 
                                       Volume.PARAMETRIC_COORDS[:,1:3], 
                                       self.points)
          
    def pointInHexa(self,p):
        """
        This is intended as a lightweigth test for checking if a point (or
        a set of them) lies inside an hexahedron. This uses the algorithm
        implemented in :mod:`Utils`.
        
        Parameters
        ----------
        p : numpy.array (N,3)
            A collection of points
            
        Returns
        -------
        result : numpy.array (N) 
            An array with "1" corresponding to points in the hexa or "0" 
            otherwise
        """
        return Utils.pointInHexa(p,self.points)    
    
    def interpolate(self, p, verify = True):
        """
        This is a convienience function to interpolate the data in the ".data"
        field of this object. As the field is not under surveillance, it is
        the responsibility of the user to ensure that the field contains a
        numpy.array with the shape (M,8), i.e. one data point for each
        vertex
        
        
        
        Parameters
        ----------
        p : numpy.array (N,3)
            A collection of points
            
        verify : boolean (True)
            Verifies if the point is contained in the hexa defined by the
            edges. If not, the corresponding row of result will be zeroed out
            
        Returns
        -------
        result : numpy.array (N,M) 
            An array with the data from the field ".data" interpolated
        """
        if not verify:
            return Utils.hexaInterpolation(p, self.points, self.data)
        else:
            return np.einsum("ij,i->ij",
                             Utils.hexaInterpolation(p, self.points, self.data),
                             Utils.pointInHexa(p,self.points))        
    

class RayBundle(Assembly):
    """
    This class represents an arbitrary number of light rays. The reason for
    having them wrapped in a single class is that it allows full vectorization
    of the raytracing procedure.
    
    This can be called a disposable class
    """
    TRACING_FOV               = 1
    TRACING_LASER_REFLECTION  = 0
    PLOTDIMS                  = -1
    def __init__(self):
        Assembly.__init__(self)
        self.name                       = 'Bundle ' + str(self._id)
        self.material                   = None
        # Ray tracing configuration
        self.maximumRayTrace            = 10
        self.stepRayTrace               = 10
        self.preAllocatedSteps          = 10
        self.wavelength                 = None
        self.startingPoints             = None
        self.initialVectors             = None
        # Ray tracing statistics
        self.rayPaths                   = None
        self.rayLastVectors             = None
        self.rayIntersections           = None
        self.rayLength                  = None
        self.steps                      = None
        self.finalIntersections         = None
        
    @property
    def bounds(self):           return None
        
    def append(self, initialVector, initialPosition = None, wavelength = 532e-9):
        """
        This method is used to append new rays in the bundle. 
        
        Parameters
        ----------
        initialVector : numpy.array (N x 3)
            if N rays are given, each element is the initial vector for ray
            tracing of each ray
        initialPosition : numpy.array (N x 3)
            if no parameter is passed, rays will depart from the origin of the
            bundle. Otherwise they will depart from the given points. If N
            points were given, a single common starting point can be given
        wavelength : numpy.array (N)
            the wavelength of the rays in meters (this changes the color and
            the behavior of the rays, if any dispersing element is present in
            the simulation
                                 
        Notes
        -----
        
        *   If the starting point is omitted, it will assume that the rays
            departs from the origin of the bundle
            
        *   This method was not made to be efficient in a loop (there are many
            checks), so it should be used ideally only once to append all rays
            
        *   This method destroys data that was already ray-traced, if any
        
        Examples
        --------        
        1) A single ray is inserted::
        
        >>> bundle = RayBundle()
        >>> bundle.append(np.array([1,0,0]), np.array([0,0,0]))
        
        2) Several rays are inserted, each with a starting point::
        
        >>> bundle.append(np.array([[1,0,0],[1,0,0]]), 
        ...               np.array([[0,0,0],[0,0,1]]))
        
        3) Several rays are inserted, with a common starting point::
        
        >>> bundle.append(np.array([[1,0,0],[0,1,0]], 
        ...               np.array([0,0,0]))
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
        This method is not implemented, and is present only to respect the 
        Assembly interface. As deleting a single ray requires many matrix 
        reshapings, it is better to clear all the data and redo the ray
        tracing. 
        
        Raises
        ------
        NotImplementedError
        """
        return NotImplementedError
    
    def clear(self):
        """
        Removes all rays and ray tracing data from the bundle.
        """
        self.wavelength                 = None
        self.startingPoints             = None
        self.initialVectors             = None
        self.clearData()

    def translateImplementation(self, translation):
        """
        This method changes the rays starting points, and waits for clear data
        to delete all ray tracing related information.
        
        Parameters
        ----------
        translation : numpy.array (3)
            A [dx, dy, dz] vector
        """
        try:
            self.startingPoints             = self.startingPoints + translation
        except TypeError:
            pass # there might be no starting points registered
               
    def rotateImplementation(self, angle, axis, pivotPoint):
        """
        This method rotates the starting points and deletes all ray tracing
        data, because a rotation or a translation of light 
        rays may completely change the light paths.
        
        Parameters
        ----------
        angle
            Angle : scalar (in radians)
        axis : numpy.array (1 x 3)
            Vector around which the rotation occurs.
        pivotPoint : numpy.array (1 x 3)
            Point in space around which the rotation occurs. If not given, 
            rotates around origin.
        """
        try:
            self.startingPoints = Utils.rotatePoints(self.startingPoints, 
                                                     angle, axis, pivotPoint)
            self.initialVectors = Utils.rotateVector(self.initialVectors,
                                                      angle, axis)
        except TypeError:
            pass
              
    def trace(self, tracingRule = TRACING_FOV, restart = False):
        """
        A method for starting ray tracing. The bundle must be included in the
        environment assembly where ray tracing will be done (the position, 
        however, is not important)
        
        Parameters
        ----------
        tracingRule
            Either RayBundle.TRACING_FOV or RayBundle.TRACING_LASER_REFLECTION,
            this parameter tells if ray tracing should stop at opaque surfaces
            (this is the case when tracing to determine the field of view) or
            not (to trace for laser safety calculations)
            
        Raises
        ------
        RuntimeError
            If the bundle is not inserted in an assembly, this error will be 
            raised.
        """
        # Make sure everything is clear
        nrays = np.size(self.startingPoints, 0)
        
        if restart:
            # Memorize scope variables
            currVector = self.rayLastVectors
            step       = self.steps
            rayPoints  = self.rayPaths
            rayIntersc = self.rayIntersections
            
            # Reallocate vectors
            rayPoints  = Utils.reallocateArray(rayPoints, 
                                               self.preAllocatedSteps)
            rayIntersc = Utils.reallocateArray(rayIntersc, 
                                               self.preAllocatedSteps)
            distance   = self.rayLength
            surfaceRef = self.finalIntersections
        else:
            self.clearData() 
            # Shortcut to the number of rays being traced:
            
            # Initialize variables
            distance                    = np.zeros(nrays)
            currVector                  = copy.deepcopy(self.initialVectors)
            self.finalIntersections     = np.empty(nrays,dtype='object')
            self.rayLastVectors         = copy.deepcopy(self.initialVectors)
            # Do matrix pre-allocation to store ray paths
            rayPoints   = np.empty((self.preAllocatedSteps, nrays, GLOBAL_NDIM),
                                    dtype='double')
            rayIntersc  = np.empty((self.preAllocatedSteps, nrays, 1),
                                    dtype='object')
            step                  = 0
            
            rayPoints[0,:,:]      = copy.deepcopy(self.startingPoints)
                 
        stepsize              = np.ones(nrays) * self.stepRayTrace
        stepsize[distance + stepsize > 
                 self.maximumRayTrace] = self.maximumRayTrace - distance[distance + stepsize > 
                                                                         self.maximumRayTrace]
        
        rayPoints[step+1,:,:] = (rayPoints[step,:,:] + 
                                 np.tile(stepsize,(GLOBAL_NDIM,1)).T*
                                 currVector)

        # Routine to find the top element in the hierarchy
        if self.parent is None:
            raise RuntimeError("Could not find parent element. " +
                    "Is this bundle really inside an assembly?") 
        topComponent = self
        while topComponent.parent is not None:
            topComponent = topComponent.parent
        # Ray tracing loop
        while np.sum(stepsize) > GLOBAL_TOL:
            
            # Increase matrix size (if this is done too often, performance is
            # really, really bad. So adjust self.preAllocatedSteps wisely
            if step + 2 >= np.size(rayPoints,0):
                rayPoints  = Utils.reallocateArray(rayPoints, 
                                                   self.preAllocatedSteps)
                rayIntersc = Utils.reallocateArray(rayIntersc, 
                                                   self.preAllocatedSteps)
              
            # Ask for the top assembly to intersect the rays with the whole
            # Component tree, will receive results only for the first 
            # intersection of each ray
            [t, coords, _, N, 
             surfaceRef]   = topComponent.intersections(rayPoints[step],
                                                        rayPoints[step+1],
                                                        GLOBAL_TOL)

            self.finalIntersections[t <= 1] = surfaceRef[t <= 1]
            rayIntersc[step+1,:,:]          = np.reshape(surfaceRef,(nrays,1))
        
            # Calculate the distance ran by the rays
            distance = (distance + 
                        t * (t <= 1) * stepsize + (t > 1) * stepsize)
        
            # Calculate the next vectors
            currVector  = self._calculateNextVectors(currVector, 
                                                     t, N, 
                                                     surfaceRef,
                                                     tracingRule)
            self.rayLastVectors[Utils.norm(currVector) > 0] = \
                                        currVector[Utils.norm(currVector) > 0]
            # Calculate next step size
#            stepsize = ((self.stepRayTrace * 
#                        (distance + self.stepRayTrace <= self.maximumRayTrace)+ 
#                        (self.maximumRayTrace - distance) * 
#                        (distance + self.stepRayTrace > self.maximumRayTrace)))#*
#                       #Utils.norm(currVector))           
            stepsize[distance + stepsize > 
                     self.maximumRayTrace] = \
                     (self.maximumRayTrace - distance)[distance + stepsize > 
                                                       self.maximumRayTrace]

            # Calculate next inputs:
            step              = step + 1
            rayPoints[step]   = coords
            rayPoints[step+1] = (rayPoints[step] +
                                 currVector * 
                                 np.tile(stepsize,(GLOBAL_NDIM,1)).T)
                   
        # Now, clean up the mess with the preallocated matrix:
        self.rayPaths                   = rayPoints[range(step+1)]
        self.rayIntersections           = rayIntersc[range(step+1)]
        # Save ray tracing statistics
        self.rayLength                  = distance
        self.steps                      = step
        self.finalIntersections         = surfaceRef
        
        # And create lines to represent the rays
        if restart:
            for n in range(nrays):
                self._items[n].points = self.rayPaths[:,n,:]
        else:
            self._items = np.empty(nrays,"object")
            for n in range(nrays):
                self._items[n]        = Line()
                self._items[n].parent = self
                self._items[n].points = self.rayPaths[:,n,:]
                self._items[n].color  = Utils.metersToRGB(self.wavelength[n])
            
    def _calculateNextVectors(self, currVector, t, N, surface, tracingRule):
        """
        A method to calculate the vectors to continue ray tracing. This includes
        the logic of determining if the tracing stops at opaque interfaces or 
        not.
        
        Parameters
        ----------
        currVector : numpy.array (N x 3)
            The current ray path
        t : numpy.array (N)
            The position of the intersection given by the equation
            p = p0 + t*(p1 - p0). The value of t must be between zero 
            (exclusive) and 1 (inclusive) to be considered valid.
        N : numpy.array (N x 3)
            The normal vector of the intersected surface
        surface : numpy.array(N) of Components
            The references to the intersected surfaces
        tracing rule : RayBundle.TRACING_FOV or TRACING_LASER_REFLECTION
            This parameter tells if ray tracing should stop at opaque surfaces
            (this is the case when tracing to determine the field of view) or
            not (to trace for laser safety calculations)
            
        Returns
        -------
        vectors : numpy.array (N x 3)
            the vectors indicating the direction that ray paths must continue
            in ray tracing
            
        Raises
        ------
        AssertionError
            If the norm(N) or norm(currVector) is not 1. 
        """
        # Returns same vector if no intersection was found
        if (t > 1 + GLOBAL_TOL).all():
            return currVector
        
        # Keep these assertions here if you are unsure that you are getting
        # the correct input data
        assert(Utils.aeq(N, Utils.normalize(N)))
        assert(Utils.aeq(currVector, Utils.normalize(currVector)))
        
        # Calculate as if all rays were reflected
        reflected = currVector - (2 * N * 
                                  np.tile(np.sum(N * currVector,1),
                                          (GLOBAL_NDIM,1)).T)
        
        # Calculate refractions
        # Important calculation:
        NdotV       = np.sum(currVector * N, 1)
        # Properties:
        # NdotV < 0 => Ray entering the surface (normals point outwards)
        # NdotV > 0 => Ray exiting the surface
        # NdotV = 0 => Should not happen, as the intersection algorithm
        #               rejects that. Spurious cases will be filtered later
        
        Nsurf           = np.zeros_like(surface)
        Nparent         = np.zeros_like(surface)
        N1              = np.ones_like(surface)
        N2              = np.ones_like(surface)
        surfaceProperty = np.zeros((len(surface),len(Component.MIRROR)),'bool')
        #cosTheta1       = -NdotV

        for n, surf in enumerate(surface):
            if surf is not None:
                Nsurf[n]   = surf.refractiveIndex(self.wavelength[n])
                Nparent[n] = surf.parent.refractiveIndex(self.wavelength[n])
                surfaceProperty[n] = surf.surfaceProperty
                                                                      
        # If entering surface, N1 is the external index of refraction, N2 is
        # the internal
        N1[(NdotV < 0) * (t <= 1)] = Nparent[(NdotV < 0) * (t <= 1)]
        N2[(NdotV < 0) * (t <= 1)] = Nsurf[  (NdotV < 0) * (t <= 1)]
        # and vice versa
        N1[(NdotV > 0) * (t <= 1)] = Nsurf[  (NdotV > 0) * (t <= 1)]
        N2[(NdotV > 0) * (t <= 1)] = Nparent[(NdotV > 0) * (t <= 1)]
            
        
        # formulation taken from: http://en.wikipedia.org/wiki/Snell's_law
        cosTheta2                 = 1 - (1 - NdotV**2) * ((N1 / N2) ** 2)
        cosTheta2[cosTheta2 >= 0] = cosTheta2[cosTheta2 >= 0]**0.5
        
        refracted = (np.tile(N1/N2,(GLOBAL_NDIM,1)).T * currVector + 
                     np.tile(np.sign(-NdotV)*
                             ((N1/N2)*np.abs(NdotV) - cosTheta2), 
                             (GLOBAL_NDIM,1)).T * 
                     N)
        refracted = Utils.normalize(refracted)
    
        #   Big if block to sort the cases out
        #=========================================
        # First, assume rays were undisturbed:
        result = currVector
        # Then substitute those who were successfully refracted / pass through
        result[(cosTheta2 <= 1 + GLOBAL_TOL) * 
               surfaceProperty[:,1]] = Utils.normalize(
                                        refracted[(cosTheta2 <= 1 + GLOBAL_TOL)* 
                                                  surfaceProperty[:,1]])
        # Then zero those rays who found a dump
        if tracingRule == RayBundle.TRACING_FOV:
            result[surfaceProperty[:,2] + 
                   surfaceProperty[:,3]] = (0 * result[surfaceProperty[:,2] + 
                                                       surfaceProperty[:,3]])
        else:
            result[surfaceProperty[:,3]] = 0 * result[surfaceProperty[:,3]]
        # Then put reflected rays
        result[surfaceProperty[:,0]] = reflected[surfaceProperty[:,0]]
        result[cosTheta2 < 0]        = reflected[cosTheta2 < 0]
        
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
    
if __name__=="__main__":
    """
    Code for unit testing basic functionality of classes in the Core module
    """
    print ""
    print "*********************************************************************"
    print "**********       pyvsim primitives module unit test      ************"
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
    assembly += part
    print "Successfully created a project tree"
    print part.name
    print assembly.name
    #
    # Test refraction coefficient
    #
    print "************     Index of refraction calculation   ******************"
    part.refractiveIndexConstant    = 1
    assert Utils.aeq(part.refractiveIndex(532e-9), 1)
    sellmeierCoeffs      = np.array([[1.03961212, 0.00600069867],
                                     [0.23179234, 0.02001791440],
                                     [1.01046945, 103.560653000]])
    part.material = Library.Glass(sellmeierCoeffs)
    
    assert Utils.aeq(part.refractiveIndex(532e-9), 1.51947, 1e-3)
    assert Utils.aeq(part.refractiveIndex(486e-9), 1.52238, 1e-3)
    assert Utils.aeq(part.refractiveIndex(np.ones(10)*532e-9),
                     np.ones(10)*1.51947, 1e-3)    
    print "************              Intersection test        ******************"
    #         hit            miss                hit                  miss
    p0 = [[0.5,0.5,-1],  [1.5,1.5,-1],  [0.999999,2 ,0.999999],[-1,-1e-6,-1e-6]]
    p1 = [[0.5,0.5, 2],  [1.5,1.5, 2],  [1       ,-1,1       ],[ 2,    0,    0]]
    t0 = [           0,            10,                       0,              10]
    p0 = np.array(p0)
    p1 = np.array(p1)
    t0 = np.array(t0)
    # Parameter for the speed test
    repetitions = 1000
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
    [t2, coords2, inds2, norms2, refs2] = assembly.intersections(p0, p1, 
                                                                 GLOBAL_TOL)
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
    [t2, coords2, inds2, norms2, refs2] = assembly.intersections(p0, p1, 
                                                                 GLOBAL_TOL)
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
    print "Reference result from pyVSim v.0 - 71500 polygon intersection/s"

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
    print assembly
    
    bundle = RayBundle()
    bundle.translate(np.array([0.5,0.5,0.5]))
    bundle.append(np.array([[1,0,0],[0,1,0],[0,0,1]]))
    assembly += bundle
    
    print "Tracing ray bundle"
    print "Pre allocated steps : ", bundle.preAllocatedSteps
    print "Step ray trace      : ", bundle.stepRayTrace
    tic.tic()
    bundle.trace(RayBundle.TRACING_LASER_REFLECTION)
    tic.toc()
    print "Ray lengths         : ", bundle.rayLength
    print "Number of steps     : ", bundle.steps
#    print bundle.rayPaths[-1]
    
    bundle.preAllocatedSteps          = 10
    bundle.stepRayTrace               = 5
    print "Tracing ray bundle"
    print "Pre allocated steps : ", bundle.preAllocatedSteps
    print "Step ray trace      : ", bundle.stepRayTrace
    
#    print bundle.startingPoints
#    print bundle.initialVectors
    bundle.rotate(np.pi/4, np.array([0,0,1]))
#    print bundle.initialVectors
    
    tic.tic()
    bundle.trace()
    tic.toc()
    print "Ray lengths         : ", bundle.rayLength
    print "Number of steps     : ", bundle.steps
#    print bundle.rayPaths[-1] 
    
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
    coords = m.parametricToPhysical(np.array([[0,0],[1,1],[-1,-1]]))
    assert Utils.aeq(coords[0], np.zeros(3))
    assert Utils.aeq(coords[1], m.parametricToPhysical(np.array([1,1])))
    assert Utils.aeq(coords[2], m.parametricToPhysical(np.array([-1,-1])))
    print m.points
    print theorypoints
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