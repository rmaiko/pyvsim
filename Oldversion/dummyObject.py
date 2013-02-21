from __future__ import division
import vtk
import numpy as np
import copy
import Utils
import vec
import math

class Object():
    def __init__(self):
        """
        All objects are initialized as non-optical objects, please refer to 
        following table for convention:
                                    mirror  lens    dump    non-optical lasersheet
        indexOfRefraction           0       !=0     0       0           0
        indexOfRefractionAmbient    0       !=0     0       0           0
        indexOfReflection           1       any*    0       -1          0
        isLightSource               0       0       0       0           any
        * Fresnel equations are not implemented
        
        """
        self.origin                     = np.array([0,0,0])
        self.x                          = np.array([1,0,0])
        self.y                          = np.array([0,1,0])
        self.z                          = np.array([0,0,1])
        self.indexOfRefraction          =  0
        self.indexOfRefractionAmbient   =  0
        self.indexOfReflection          = -1
        self.isLightSource              =  0
        self.points                     = np.array([])
        self.connectivity               = np.array([])
        self.normals                    = None
        self.color                      = None
        self.opacity                    = 0.5
        # Variables for raytracing
        self.bounds                             = None
        self.triangleVectors                    = None
        self.trianglePoints                     = None
        self.triangleNormals                    = None
        
    def intersectLineWithPolygon(self,p0,p1,tol=1e-7):
        """ Method for finding all intersections between polygon and segment
        Inputs:
        p0 - segment initial point - 3 component list
        p1 - segment final point   - 3 component list
        
        The result is a dictionary of dictionaries, being the primary key the
        parameter t, so that the intersection point p is:
        p = p0 + t*p1
        
        Then, the dictionary is:
        { t1 : {'coords' : [x,y,z]       #coordinates of the intersection
                'cellno' : int           #intercepted triangle number
                'normal' : [vx,vy,vz]}   #normal vector at the point
                'n'      : double        #index of refraction
                'n_ext'  : double        #index of refraction of the ambient
                'r'      : double        #reflection coefficient (-1, 0 or 1)
          t2 : ...
          ...
          tn : ...
         }
         
        A way to use this dictionary is to use the method keys and sort them in
        order to find the first intersection:
        k = dict.keys()
        k.sort()
        #the first intersection is then:
        dict[k[0]]
         
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
        # Call fast test (bounding box test)
        #
#        if self.intersectWithBoundingBox(p0,p1) == 0:
#            return None
        #
        # Start dumb search, step 1 - determine if line intercept triangle plane
        #
        try:
            Ptriangles                  = self.trianglePoints
            [V1,V2,UU,UV,VV,UVden]      = self.triangleVectors
            [N,Nnorm]                   = self.triangleNormals
        except:
            Ptriangles      = self.points[self.connectivity]
            V1              = Ptriangles[:,1] - Ptriangles[:,0]
            V2              = Ptriangles[:,2] - Ptriangles[:,0]
            N               = np.cross(V1,V2)
            UU              = vec.dot(V1,V1)
            UV              = vec.dot(V1,V2)
            VV              = vec.dot(V2,V2)
            UVden           = (UV**2 - UU*VV)
            self.triangleVectors    = [V1,V2,UU,UV,VV,UVden]
            self.trianglePoints     = Ptriangles
            self.triangleNormals    = [N,vec.normalize(N)]
  
        nlines = np.size(p1,0)
        ntris  = np.size(N,0)
        #den = vec.dot(N,(p1-p0))       # equivalent to dot for each vector 
        v_temp = p1-p0
        v_temp = v_temp.reshape(nlines,1,3)
        den = np.sum(N*v_temp,2)
        # den size is:
        #     rows    : n_lines
        #     columns : n_tris
        
        
        
        # When denominator is zero, the line is either parallel or contained in
        # the triangle plane. In order to make it easier, will throw this case
        # away.
        #
        # Substituting zeros to np.inf yields t=0, a case that is further ignored
        # and dealing with infinity seems faster than NaN for numpy
        den[(den == 0)] = np.inf

        # Calculates parameter for all triangles
        v_temp = Ptriangles[:,0] - p0.reshape(nlines,1,3)
        # this is np.dot(N, Ptriangles[:,0] - p0) / den
        # it means we're finding the parameters T to find the line-plane 
        # intersection
        T  = np.sum(N * v_temp,2) / den 
        # T size is:
        #     rows    : n_lines
        #     columns : n_tris
                
        # This equation is basically p = p0 + t*(p1-p0)
        # this is to find the intersection point between line and triangle plane
        v_temp = p1-p0
        v_temp.reshape(nlines,1,3)
        T.resize(nlines,ntris,1)
        np.tile(T.T,(1,1,3))
        print T
        P = p0.reshape(nlines,1,3) + T * v_temp
        
        V       = P - Ptriangles[:,0]
        #UW      = vec.dot(V1,V)
        UW      = np.sum(V1 * V,2)
        #VW      = vec.dot(V2,V)  
        VW      = np.sum(V2 * V,2)

        S1 = (UV*VW - VV*UW) / UVden
        T1 = (UV*UW - UU*VW) / UVden

        # Analyzes intersection conditions and create a list with all
        # intersections
        # first line : check if intersection is not out of segment
        # second line: if intersection is not "behind" the origin of vectors
        # third line : if intersection is not "beyond" vectors
        #
        # behind      V2
        #        o----------->
        #         \  in     /
        #          \       /  beyond
        #        V1 \     /
        #            \   /
        #             \ /
        #              v 
        Tindexes = np.nonzero((T > tol)*(T < 1+tol)* \
                              (S1 > -tol)*(T1 > -tol)* \
                              (S1 + T1 <= 1 + tol))[0] 
        
        if len(Tindexes) > 0:
            result = {}
            for i in Tindexes:
                # DEPRECATED:
                # result[T[i]] = {'coords'  :P[i],
                                # 'cellno'  :ints[i],
                                # 'normal'  :self.__computeNormal(ints[i],P[i]),
                                # 'n'       :self.indexOfRefraction,
                                # 'n_ext'   :self.indexOfRefractionAmbient,
                                # 'r'       :self.indexOfReflection,
                                # 'issource':self.isLightSource}
                result[T[i]] = {'coords'  :P[i],
                                'cellno'  :i,
                                'normal'  :self.__computeNormal(i,P[i]),
                                'n'       :self.indexOfRefraction,
                                'n_ext'   :self.indexOfRefractionAmbient,
                                'r'       :self.indexOfReflection,
                                'issource':self.isLightSource}
            return result
        else:
            return None
        
    def getBounds(self):
        """
        Returns [xmin,xmax,ymin,ymax,zmin,zmax] or an array of zeros if the
        geometry is not defined
        """
        if len(self.points) > 0:
            xmin = [min(self.points[:,0]),min(self.points[:,1]),min(self.points[:,2])]
            xmax = [max(self.points[:,0]),max(self.points[:,1]),max(self.points[:,2])]
            self.bounds = [xmin,xmax]
            return np.array([min(self.points[:,0]),max(self.points[:,0]),  #xmin, xmax
                             min(self.points[:,1]),max(self.points[:,1]),  #ymin, ymax
                             min(self.points[:,2]),max(self.points[:,2])]) #zmin, zmax
        else:
            return np.zeros(6)
        
    def intersectWithBoundingBox(self,p0,p1):
        """ Determines if the line segment intersects the box bounding the polygon
        Inputs:
        p0 - segment initial point - 3 element list
        p1 - segment final point   - 3 element list
        
        Returns:
        if intersection     - 1
        if not intersection - 0
        
        The algorithm implemented here was taken almost verbatim from:
        author = {Amy Williams and Steve Barrus and R. Keith and Morley Peter 
        Shirley},
        title = {An efficient and robust ray-box intersection algorithm},
        journal = {Journal of Graphics Tools},
        year = {2003},
        volume = {10},
        pages = {54}
        
        There is one problem in:
        if (p1[d]-p0[d]) == 0:
                divx = np.inf
        In fact, the code should verify the sign of zero and apply to infinity, 
        but I could not find a way in python to do so. This seems not to affect 
        the result too much (this code also does not need this high precision 
        ray tracing)
        """
        try:
            [xmin,xmax] =  self.bounds
        except:
            self.getBounds()
            [xmin,xmax] =  self.bounds

        tmin = [0,0,0]
        tmax = [0,0,0]
        
        for d in range(3):
            if (p1[d]-p0[d]) == 0.0:
                divx = np.inf
            else:
                divx = 1.0 / (p1[d]-p0[d])
                
            if (p1[d]-p0[d] >= 0):
                tmin[d] = (xmin[d] - p0[d])*divx
                tmax[d] = (xmax[d] - p0[d])*divx
            else:
                tmin[d] = (xmax[d] - p0[d])*divx
                tmax[d] = (xmin[d] - p0[d])*divx

            
        if ((tmin[0] > tmax[1]) or (tmin[1] > tmax[0])):
            return 0
            
        if (tmin[1] > tmin[0]):
            tmin[0] = tmin[1]
        if (tmax[1] < tmax[0]):
            tmax[0] = tmax[1] 
            
        if ((tmin[0] > tmax[2]) or (tmin[2] > tmax[0])):
            return 0
            
        if (tmin[2] > tmin[0]):
            tmin[0] = tmin[2]
        if (tmax[2] < tmax[0]):
            tmax[0] = tmax[2]
            
        return ((tmin[0] < 1) * (tmax[0] > 0))    
        
    def __computeNormal(self,n,p):
        """ Compute normal vector given the triangle number and the intersecting point
        Inputs:
        n - number of the triangle  - int
        p - intersecting point      - 3-element list
        
        This method returns a 3-element list corresponding to the normalized normal
        vector. There are two algorithms implemented:
        
        1) If polygon has no normal vector information:
            returns vector normal to the triangle, this will result in a faceted
            appearance of the object - use this if representing something planar,
            faceted, opaque, etc...
        2) If polygon has one normal per vertice
            returns interpolation (using barycentric coordinates) of the normals
            on the triangle vertices - use this if representing lenses, etc
            
        WARNING - will return a result even if point is not on the polygon
                      
        """
        if self.normals is None:
            # normal = Utils.triangleNormal(pcoords[0],pcoords[1],pcoords[2])
            try:
                normal = self.triangleNormals[1][n]
            except:
                Ptriangles      = self.points[self.connectivity]
                V1              = Ptriangles[:,1] - Ptriangles[:,0]
                V2              = Ptriangles[:,2] - Ptriangles[:,0]
                N               = np.cross(V1,V2)
                self.triangleNormals    = [N,vec.normalize(N)]
                normal = self.triangleNormals[1][n]
        else:
            pcoords = self.points[self.connectivity[n]]
            lambdas = Utils.barycentricCoordinates(p,pcoords[0],pcoords[1],pcoords[2])
            norms = self.normals[self.connectivity[n]]
            # Baricentric interpolation
            normal = lambdas[0]*np.array(norms[0]) + \
                     lambdas[1]*np.array(norms[1]) + \
                     lambdas[2]*np.array(norms[2])
            normal = normal / np.linalg.norm(normal)         
 
        return normal
       
    def translate(self,v):
        """
        This method should be used when there is a change in lasersheet position,
        it prevents the creation of inconsistent vectors or plotting
        
        Inputs
        
        v -  vector to translate the lasersheet
        
        """
        self.origin = self.origin + v
        if self.points is not None:
            if len(self.points) > 0:
                self.points = self.points + v
        self.destroyData()
        
    def rotateAroundAxis(self,angle,axis,pivotPoint=None):
        """
        Rotates all points of the object around "origin" and through "axis"
        """
        if pivotPoint is None:
            pivotPoint = self.origin
        self.origin     = Utils.rotatePoints(self.origin,angle,axis,pivotPoint)
        self.x          = Utils.rotateVector(self.x,angle,axis)
        self.y          = Utils.rotateVector(self.y,angle,axis)
        self.z          = Utils.rotateVector(self.z,angle,axis)
        self.points     = Utils.rotatePoints(self.points,angle,axis,pivotPoint)
        self.destroyData()       
        
    def alignTo(self,direction,normal,lateral=None):
        """
        This method allows the alignment of the object to a specific direction
        given by a vector
        
        v - vector pointing to the desired lasersheet direction
        
        """
        x2 = direction
        y2 = normal
        z2 = lateral
        if normal is not None and lateral is None:
            z2 = np.cross(direction,normal)
        if normal is None and lateral is not None:
            y2 = np.cross(lateral,direction)
        
        x2 = vec.normalize(x2)
        y2 = vec.normalize(y2)
        z2 = vec.normalize(z2)
      
        Xnew = np.vstack([x2,y2,z2])
        Xold = np.array([self.x,
                         self.y,
                         self.z])
        M = np.linalg.solve(Xold,Xnew)
        assert np.linalg.det(M) - 1 < 1e-6
        self.x = x2
        self.y = y2
        self.z = z2
        # self.origin = np.dot(self.origin,M)
        if self.points is not None:
            if len(self.points) > 0:
                self.points = np.dot(self.points,M)       
        self.destroyData()
        
    def destroyData(self):
        """
        Implement this method whenever your object possesses geometrical features
        that are calculated from their interaction with the ambient (e.g. - any
        raytraced features). This method is called for all spatial transformations
        """
        self.bounds                             = None
        self.triangleVectors                    = None
        self.trianglePoints                     = None
        self.triangleNormals                    = None


  
if __name__=="__main__":
    """
    Code for unit testing basic functionality of class
    """
    import Utils
    from pprint import pprint
    
    tic = Utils.Tictoc()
    
    points = [[0,0,0],
              [1,0,0],
              [1,1,0],
              [0,1,0],
              [0,0,1],
              [1,0,1],
              [1,1,1],
              [0,1,1]]

    # mixed normals
    # conn = [[0,1,3],[1,2,3], #xy-plane +z inwards
            # [0,1,4],[1,5,4], #xz-plane -y outwards
            # [4,5,7],[5,6,7], #xy-plane +z outwards
            # [3,6,7],[3,2,6], #xz-plane -y inwards
            # [1,5,6],[1,6,2], #yz-plane -x inwards
            # [0,3,7],[0,7,4]] #yz-plane +x inwards

    # normals pointing outside
    # conn = [[5,7,4],[5,6,7], # normal +z
           # [3,2,1],[0,3,1], # normal -z
           # [3,6,2],[6,3,7], # normal +y
           # [1,5,4],[4,0,1], # normal -y
           # [5,1,6],[1,2,6], # normal +x
           # [7,0,4],[7,3,0]] # normal -x

    # normals pointing inside
    conn = [[4,7,5],[7,6,5], # normal -z
            [1,2,3],[1,3,0], # normal +z
            [2,6,3],[7,3,6], # normal -y
            [4,5,1],[1,0,4], # normal +y
            [6,1,5],[6,2,1], # normal -x
            [4,0,7],[0,3,7]] # normal +x
                       
    mesh = Object()
    mesh.points = np.array(points)
    mesh.connectivity = np.array(conn)
    
    # ===========================================================
    # Normal play
    #
    #
    # Edge normals
    #
    normals = []
    for n in range(8):
        norm = mesh.points[n] - np.array([0.5,0.5,0.5])
        norm = -norm/np.sqrt(np.dot(norm,norm))
        normals.append(norm)

    # # ===========================================================
    # # normal calculation and display
    # #
    # print "###                     NORMALS                   ###"
    # centerpoints = [[0.499999999,0.4999999999,0.000000000],
                    # [0.500000001,0.5000000001,0.000000000],
                    # [0.499999999,0.0000000000,0.499999999],
                    # [0.500000001,0.0000000000,0.500000001],
                    # [0.000000000,0.5000000001,0.500000001],
                    # [0.000000000,0.4999999999,0.499999999],
                    # [0.499999999,0.4999999999,1.000000000],
                    # [0.500000001,0.5000000001,1.000000000],
                    # [0.499999999,1.0000000000,0.499999999],
                    # [0.500000001,1.0000000000,0.500000001],
                    # [1.000000000,0.5000000001,0.500000001],
                    # [1.000000000,0.4999999999,0.499999999]]
    # #
    # # Test using implicit normals (same vertice as triangle)
    # #   
    # print "Implicit normals:"
    # tic.reset()
    # for n in range(12):
        # result = None
        # for p in centerpoints:
            # pts  = mesh.points[mesh.connectivity[n]]
            # lambdas = Utils.barycentricCoordinates(p,pts[0],pts[1],pts[2])
            
            # temp = mesh.__computeNormal(n,p)
            # if (sum(lambdas) - 1)**2 < 1e-6:
                # assert (((temp[0]**2 - 1)**2 < 1e-6) or \
                       # ((temp[1]**2 - 1)**2 < 1e-6) or \
                       # ((temp[2]**2 - 1)**2 < 1e-6))
                # result = temp
        # assert result is not None   
    # tic.toc()
    # #
    # # Test using normals on vertices
    # #
    # mesh.normals = np.array(normals)
    # print "Normals on vertices:"
    # tic.reset()
    # for n in range(12):
        # result = None
        # for p in centerpoints:
            # pts  = mesh.points[mesh.connectivity[n]]
            # lambdas = Utils.barycentricCoordinates(p,pts[0],pts[1],pts[2])
            
            # temp = mesh.__computeNormal(n,p)
            # if (sum(lambdas) - 1)**2 < 1e-6:
                # assert (((temp[0]**2 - 1)**2 < 1e-6) or \
                       # ((temp[1]**2 - 1)**2 < 1e-6) or \
                       # ((temp[2]**2 - 1)**2 < 1e-6))
                # result = temp
        # assert result is not None    
    # tic.toc()
    # print "###                  END NORMALS                  ###"    
    
    # ===========================================================
    # Intersection with bounding box test
    # 
    # tic.reset()
    # print "###         INTERSECTIONS WITH BOUNDING BOX       ###" 
    # # obvious hit
    # assert mesh.intersectWithBoundingBox(np.array([0.5,0.5,-1]),
                                         # np.array([0.5,0.5,2]))  is not 0
    # # obvious miss    
    # assert mesh.intersectWithBoundingBox(np.array([1.5,1.5,-1]),
                                         # np.array([1.5,1.5,2]))       is 0
    # # close hit    
    # assert mesh.intersectWithBoundingBox(np.array([0.999999,2,1]),
                                         # np.array([1,-1,1]))        is not 0
    # # close miss    
    # assert mesh.intersectWithBoundingBox(np.array([-1,-1e-6,-1e-6]),
                                         # np.array([2,0,0]))       is 0
    # tic.toc()
    # print "###               END INTERSECTIONS                ###"

    # ============================================================
    # ===========================================================
    # Intersection with polygon box test (copied code, as the polygon is it own
    # bounding box)
    # 
    print "###          INTERSECTIONS WITH TRIANGLES          ###" 
    tic.reset()
    # # obvious hit
    # assert mesh.intersectLineWithPolygon(np.array([0.5,0.5,-1]),
                                         # np.array([0.5,0.5,2]))  is not None
    # # obvious miss    
    # assert mesh.intersectLineWithPolygon(np.array([1.5,1.5,-1]),
                                         # np.array([1.5,1.5,2]))       is None
    # # close hit    
    # assert mesh.intersectLineWithPolygon(np.array([0.999999,2,1]),
                                         # np.array([1,-1,1]))        is not None
    # # close miss    
    # assert mesh.intersectLineWithPolygon(np.array([-1,-1e-6,-1e-6]),
                                         # np.array([2,0,0]))       is None
    p0 = np.array([[0.5,0.5,-1],[1.5,1.5,-1],[0.999999,2,1],[-1,-1e-6,-1e-6]])
    p1 = np.array([[0.5,0.5,2],[1.5,1.5,2],[1,-1,1],[2,0,0]])
    mesh.intersectLineWithPolygon(p0,p1)
    tic.toc()
    print "###                END INTERSECTIONS               ###"

    # ============================================================
    
    mesh.rotateAroundAxis(45,np.array([0,0,1]))
    mesh.rotateAroundAxis(45,np.array([0,1,0]))
    Utils.displayScenario([mesh])