from __future__ import division

if __name__=="__main__":
    import numpy as np
    import Utils
    import ScatteringFunctions
    from Lasersheet import Lasersheet
    from Camera import Camera
    from ParticleCloud import ParticleCloud
    
    print ""
    
    l = Lasersheet()
    l.rotateAroundAxis(90,l.x)
    l.usefulLength               = 2
    l.intensityMultiplier        = 3e3   # J / m^2
    l.thicknessDivergence        = 0.5
    l.planeDivergence            = 15
    l.calculateVectors()
    
    p = ParticleCloud()
    p.scatteringFunction = ScatteringFunctions.mieScatteringUniform
    #p.scatteringFunction = ScatteringFunctions.arbitraryScattering
    p.setBounds([0.9,1.0,-0.05,0.05,-0.0025,0.0025])
    #p.seed(1e6)
    #p.seedLena()
    p.seedUniform(25,25,1,1e-6)
    
    l.trace([])
    p.getIllumination([l])

    c1 = Camera()
    c1.objective.translate(np.array([0.0265,0,0]))
    c1.translate(np.array([1.9,0,1]))
    c1.rotateAroundAxis(135,c1.y)
    c1.objective.rotateAroundAxis(-3.75,np.array([0,1,0]))
    c1.objective.distortionParameters       = np.array([5.5,10,20,1])

    c2 = Camera()
    c2.objective.translate(np.array([0.0265,0,0]))    
    c2.translate(np.array([0,0,1]))
    c2.rotateAroundAxis(45,c1.y)
    c2.objective.rotateAroundAxis(3.75,np.array([0,1,0]))    
    
    c1.maximumImagingDistance            = 2
    c1.objective.aperture                = 22
    c1.objective.nominalFocusDistance    = 1.38
    
    c2.maximumImagingDistance            = 2
    c2.objective.aperture                = 22
    c2.objective.nominalFocusDistance    = 1.38

    # print "Camera ", c.origin, c.x, c.y
    # print "Sensor ", c.sensor.origin, c.sensor.x, c.sensor.y
    
    c1.calculateDoF([l])
    c1.calculateMapping([l])   
    c1.recordParticleCloud(p)
    
    c2.calculateDoF([l])
    c2.calculateMapping([l])   
    c2.recordParticleCloud(p)
    
    sensorread = c1.sensor.readSensor()
    print "Sensor 1 max", sensorread.max()
    print "Sensor 1 min", sensorread.min() 
    
    sensorread = c2.sensor.readSensor()
    print "Sensor 2 max", sensorread.max()
    print "Sensor 2 min", sensorread.min() 
    
    Utils.displayScenario([l,p,c1,c2],False,False)
    # c1.sensor.displaySensor()
    # c2.sensor.displaySensor()
    
    import matplotlib.pyplot as plt 
    
    ax = plt.subplot(1,2,1)
    imgplot1 = ax.imshow(c1.sensor.readSensor()/(-1+2**c1.sensor.bitDepth))
    imgplot1.set_cmap('jet')
    imgplot1.set_interpolation('none')
    
    ax = plt.subplot(1,2,2)
    imgplot2 = ax.imshow(c2.sensor.readSensor()/(-1+2**c2.sensor.bitDepth))
    imgplot2.set_cmap('jet')
    imgplot2.set_interpolation('none')
    plt.show() 