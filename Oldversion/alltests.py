if __name__=="__main__":
    with file('vec.py','rU') as f:
        co = compile(f.read(),'foobar','exec')
        exec co in {'__name__':'__main__'}
        
    with file('Utils.py','rU') as f:
        co = compile(f.read(),'foobar','exec')
        exec co in {'__name__':'__main__'}
        
    with file('Ray.py','rU') as f:
        co = compile(f.read(),'foobar','exec')
        exec co in {'__name__':'__main__'}

    with file('Object.py','rU') as f:
        co = compile(f.read(),'foobar','exec')
        exec co in {'__name__':'__main__'}
        
    with file('Planes.py','rU') as f:
        co = compile(f.read(),'foobar','exec')
        exec co in {'__name__':'__main__'}
        
    with file('Lasersheet.py','rU') as f:
        co = compile(f.read(),'foobar','exec')
        exec co in {'__name__':'__main__'}
        
    with file('ScatteringFunctions.py','rU') as f:
        co = compile(f.read(),'foobar','exec')
        exec co in {'__name__':'__main__'}
        
    with file('Camera.py','rU') as f:
        co = compile(f.read(),'foobar','exec')
        exec co in {'__name__':'__main__'}
        
    with file('test.py','rU') as f:
        co = compile(f.read(),'foobar','exec')
        exec co in {'__name__':'__main__'}