class test(object):
    outprop = 0
    def __init__(self):
        self.__inprop = 0
        test.outprop += 1
    @property
    def inprop(self):
        return self.__inprop

    def testmethod(self):
        print "My inprop is ", self.inprop
        self.__inprop = 2
        print "Now it is ", self.inprop
        
if __name__ == '__main__':        
    t1 = test()   
    t2 = test()
    print "t1 inprop ", t1.inprop, " t2 inprop ", t2.inprop
    t1.testmethod()

