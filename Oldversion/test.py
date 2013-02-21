from __future__ import division
   
if __name__=="__main__":
    """
    Demo of STLReader
    """
    
    import vtk
    import Object
    from Lasersheet import Lasersheet
    import Utils
    import pprint  
    import numpy as np
    from Ray import *
    import vec
       
    tic = Utils.Tictoc()
        
    tic.reset()
    ding = Utils.readSTL('test.stl')
    print ""
    print "Time to read %i polys" % len(ding.connectivity)
    tic.toctask(len(ding.connectivity))

    ding.indexOfRefraction            = 2
    ding.indexOfRefractionAmbient     = 1
    ding.indexOfReflection            = 0
    ding.isLightSource                = 0
    
    ding.rotateAroundAxis(10,np.array([0,0,1]))
    ding.translate(np.array([0,-20,0]))
    
    l = Lasersheet()
    l.translate(np.array([0,50,0]))    
    l.alignTo(np.array([0,-1,0]),np.array([0,0,1]))
    l.maximumRayTrace       = 500
    l.usefulLength          = 200
    l.setDivergences(15,15)
    l.raysInPlane           = 25
    l.raysInThickness       = 25
    l.displayReflections    = True
    
    print "Time to trace the 4 main rays"
    tic.reset()
    l.trace([ding])
    tic.toc()
    print "Time to trace %i rays" % (l.raysInPlane*l.raysInThickness)
    tic.reset()
    l.traceReflections([ding])
    tic.toctask(l.raysInPlane*l.raysInThickness)
    
    rlist = [ding,l]  
  
    Utils.displayScenario(rlist,True,True)
    
    
    # tic.reset()
    # import cPickle
    # lala = {'a':1,'b':2}
    # f = open( "save.dat", "w" )
    # cPickle.dump(rlist,f)
    # f.close()
    # tic.toc()
    # print "Pickled!"
