from __future__ import division
import numpy as np
from pprint import pprint

def dot(a,b):
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
        error(a,b)
  
def linspace(a,b,n):
    """
    Just to correct an unexpected behavior in case of n = 1, where it would
    be better if it returned the midpoint between a and b
    """
    if n == 1:
        return np.array([0.5*(a+b)])
    else:
        return np.linspace(a,b,n)
  
def error(a,b):
    pprint(a) 
    pprint(b)
    raise ValueError("dimensionality error")
    
def normalize(a):
    if a.ndim == 1:
        return a / np.linalg.norm(a)
    else:
        return listTimesVec(np.sqrt(np.sum(a**2,1))**-1,a)
        
def norm(a):
    if a.ndim <= 1:
        return np.linalg.norm(a)
    else:
        return np.sqrt(np.sum(a**2,1))
        
def listTimesVec(l,v):
    try:
        if v.ndim > 1:
            return np.tile(l,(len(v[0]),1)).T * v
        else:
            return np.tile(l,(len(v),1)).T * np.tile(v,(len(l),1))
    except:
        return l*v
                 
if __name__=="__main__":
    """
    Code for unit testing vec class
    """
    import Utils

    matrix_ones     = np.ones((100,100))
    list_ones       = np.ones((100,1))
    veclist_ones    = np.ones((100,3))
   
    matrix_random   = np.random.rand(100,100)
    list_random     = np.random.rand(100)
    veclist_random  = np.random.rand(100,3)
    
    tic = Utils.Tictoc()
    
    # Testing of dot function
    print "Testing norm function performance"
    tic.reset()
    r1 = norm(list_random)
    r2 = norm(veclist_random)
    tic.toc()
    print "Testing norm function behavior"
    assert (r1 - np.sqrt(np.sum(list_random**2))) < 1e-6
    assert np.linalg.norm(r2 - np.sqrt(np.sum(veclist_random**2,1))) < 1e-6
        
    # Testing of listTimesVec function
    print "Testing listTimesVec function performance"
    tic.reset()
    r1 = listTimesVec(list_random,veclist_ones)
    r2 = listTimesVec(list_random,np.array([1,1,1]))
    tic.toc()
    print "Testing listTimesVec function behavior"
    assert np.linalg.norm(r1 - r2) < 1e-6

    
    # Testing of dot function
    print "Testing dot function performance"
    tic.reset()
    r1 = dot(matrix_random,matrix_random)
    r2 = dot(np.array([0,0,0]),veclist_random)
    r3 = dot(matrix_ones,matrix_ones)
    tic.toc()
    print "Testing dot function behavior"
    assert r1.ndim == 1
    assert np.linalg.norm(r1 - np.sum(matrix_random**2,1)) < 1e-6
    
    assert r2.ndim == 1
    assert len(r2) == 100
    assert np.linalg.norm(r2) < 1e-6
    
    assert r3.ndim == 1
    assert len(r3) == 100
    assert np.linalg.norm(r3 - 100) < 1e-6
    print "ok"
    
    # Testing of normalize function
    angle   = np.linspace(0,np.pi,500)
    vecs    = []
    vectors = []
    for a in angle:
        vecs.append([np.sin(a),np.cos(a),0])
        vectors.append((np.random.rand()+1)*np.array([np.sin(a),np.cos(a),0]))
    vecs = np.array(vecs)
    vectors = np.array(vectors)
    print "Testing normalize function performance"
    tic.reset()
    r1 = normalize(vectors[0])
    r2 = normalize(vectors)
    tic.toc()
    print "Testing normalize function behavior"
    assert np.linalg.norm(r1 - vecs[0]) < 1e-6
    assert np.linalg.norm(r2 - vecs) < 1e-6
    print "ok"
    