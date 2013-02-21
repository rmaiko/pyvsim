from __future__ import division

if __name__=="__main__":
    import numpy as np
    import Utils
    import ScatteringFunctions
    import Planes
    from Lasersheet import Lasersheet
    from Camera import Camera
    from ParticleCloud import ParticleCloud
    
    c = Camera()
    #c.objective.translate(np.array([0.0265,0,0]))
    #c.translate(np.array([1.9,0,1]))
    #c.rotateAroundAxis(135,c1.y)
    #c.objective.rotateAroundAxis(-3.75,np.array([0,1,0]))
    #c.objective.distortionParameters       = np.array([5.5,10,20,1])
    
    p = Planes.Lena()
    p.translate(np.array([6,0,0]))
    p.rotateAroundAxis(180,p.x)
    
    c.calculateDoF([p])
    c.calculateMapping([p])   
    
    Utils.displayScenario([c,p])
    
    c.recordData(p,5)
    
    import matplotlib.pyplot as plt 
    
    ax = plt.subplot(1,1,1)
    imgplot1 = ax.imshow(c.sensor.virtualData[0])
    imgplot1.set_cmap('jet')
    imgplot1.set_interpolation('none')
    plt.show() 
    