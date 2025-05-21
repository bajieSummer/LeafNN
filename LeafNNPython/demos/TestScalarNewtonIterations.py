import demoInit
from LeafNN.ConvexOptimizer.ScalarNewtonIteration import BaseScalarNewtonIteration as BSN
from LeafNN.utils.Log import Log
from LeafNN.ConvexOptimizer.ScalarNewtonIteration import ScalarNewtonIteration as SN

def calPoly3F(X,a,b,c,d):
    # y = a*x^3+b*x^2+c*x+d
    y = 0
    x2 = None 
    x3 = None
    if d!=0:y+=d
    if c!=0: y+=c*X 
    if b!=0:
        if x2 is None: x2 = X*X 
        y+=b*x2
    if a!=0:
        if x2 is None: x2 = X*X 
        if x3 is None: x3 =x2*X 
        y+=a*x3 
    return y

def calPoly3FAndGrad(X,a,b,c,d):
    # y = a*x^3+b*x^2+c*x+d
    # grad = 3*a*x^2 + 2*b*x + c
    y = 0
    grad = 0
    x2 = None 
    x3 = None
    if d!=0:y+=d
    if c!=0: 
        y+=c*X 
        grad +=c 
    if b!=0:
        if x2 is None: x2 = X*X 
        y+=b*x2
        grad+=2*b*X
    if a!=0:
        if x2 is None: x2 = X*X 
        if x3 is None: x3 =x2*X 
        y+=a*x3 
        grad +=3*a*x2 
    return (y,grad)

 
def calPolyF(X,argsList):
    # f = argsList[0] +  argsList[1]*X  + argsList[2]*X*X + argsList[3]*X*X*X + ..
    cacheX = [1]
    f = 0
    N = len(argsList)
    for i in range(N):
        if i >0:
            cacheX.append(X*cacheX[i-1])
        f+=argsList[i]*cacheX[i]
    return f

def calPolyFAndGrad(X,argsList):
    cacheX = [1]
    f = 0
    grad = 0
    N = len(argsList)
    for i in range(N):
        if i >0:
            cacheX.append(X*cacheX[i-1])
            grad+=argsList[i]*i*cacheX[i-1]
        f+=argsList[i]*cacheX[i]
    return (f,grad)

    

def testBaseScalarNewton1():
    #  y =x^2-5x+6 a=0,b=1 c=-5,d=6
    a,b,c,d=0,1,-5,6
    bsn = BSN(calPoly3FAndGrad)
    initX = -10
    (x,fx,gradient) = bsn.calRoot(initX,a,b,c,d)
    Log.Debug("testBaseScalarNewton",f"x={x},fx={fx},gradient={gradient}\n")
    initX = 10
    (x,fx,gradient) = bsn.calRoot(initX,a,b,c,d)
    Log.Debug("testBaseScalarNewton",f"x={x},fx={fx},gradient={gradient}\n")

def testBaseScalarNewton2():
    # repeated roots
    # y = a*x^3+b*x^2+c*x+d
     #  y =x^2 a=0,b=1 c=0,d=0
    #a,b,c,d=0,1,0,0
    #y =x^2 a=0,b=1 c=-6,d=9
    a,b,c,d=0,1,-6,9
    bsn = BSN(calPoly3FAndGrad)
    initX = -10
    (x,fx,gradient) = bsn.calRoot(initX,a,b,c,d)
    Log.Debug("testBaseScalarNewton",f"x={x},fx={fx},gradient={gradient}\n")
    # initX = 10
    # (x,fx,gradient) = bsn.calRoot(initX,a,b,c,d)
    # Log.Debug("testBaseScalarNewton",f"x={x},fx={fx},gradient={gradient}\n")

def testBaseScalarNewton3():
    # cubic oscillate
    # y = a*x^3+b*x^2+c*x+d
     #  y =x^3-2*x+2 a=1,b=0 c=-2,d=2
     # will be oscillated durning near (-0.5,1.8)
    a,b,c,d=1,0,-2,2
    bsn = BSN(calPoly3FAndGrad)
    initX =10# 0.99#-10,3
    (x,fx,gradient) = bsn.calRoot(initX,a,b,c,d)
    Log.Debug("testBaseScalarNewton",f"x={x},fx={fx},gradient={gradient}\n")


def testScalarNewton1():
    # simple test
     #  y =x^2-5x+6 a=0,b=1 c=-5,d=6
    a,b,c,d=0,1,-5,6
    sn = SN(calPoly3F,calPoly3FAndGrad)
    initX =0# 0.99#-10,3
    (x,fx,gradient) = sn.calRoot(initX,a,b,c,d)
    Log.Debug("testBaseScalarNewton",f"x={x},fx={fx},gradient={gradient}\n")

def testScalarNewton2():
    # cubic oscillate with lineSearch
    # y = a*x^3+b*x^2+c*x+d
     #  y =x^3-2*x+2 a=1,b=0 c=-2,d=2
     # will be oscillated durning near (-0.5,1.8)
    a,b,c,d=1,0,-2,2
    sn = SN(calPoly3F,calPoly3FAndGrad)
    initX =-0.5# 0.99#-10,3
    (x,fx,gradient) = sn.calRoot(initX,a,b,c,d)
    Log.Debug("testBaseScalarNewton",f"x={x},fx={fx},gradient={gradient}\n")

def testScalarNewton3Sadd():
    # saddle point function
    # y = x^3
    #args=[0,0,0,1]
    # y= x^4
    #argsList=[0,0,0,0,1]
    # y = x^4 + 2*x^3 + 1
    argsList = [1,0,0,2,1]
    #sn = SN(calPoly3F,calPoly3FAndGrad)
    sn = SN(calPolyF,calPolyFAndGrad)
    initX = 0.0
    (x,fx,gradient) = sn.calRoot(initX,argsList)
    Log.Debug("testBaseScalarNewton",f"x={x},fx={fx},gradient={gradient}\n")

from LeafNN.ConvexOptimizer.ScalarLineSearcher import ArmijoLineSearcher
def testScalarNewton4Armijo():
    # saddle point function
    # y = x^3
    #args=[0,0,0,1]
    #y= x^4
    #argsList=[0,0,0,0,1]
    # y = x^4 + 2*x^3 + 1
    #argsList = [1,0,0,2,1]
    # y = X^3 -2*x+2
    #argsList = [2,-2,0,1]
    # y = x^2-5x+6
    #argsList = [6,-5,1]
    # y = x^2
    argsList = [0,0,1]
    LS = ArmijoLineSearcher(calPolyF,calPolyFAndGrad)
    LS.setSigma(0.5)
    #LS = None
    sn = SN(calPolyF,calPolyFAndGrad)
    initX = 1.0
    (x,fx,gradient) = sn.calRoot(initX,argsList,lineSearcher=LS)
    Log.Debug("testBaseScalarNewton",f"x={x},fx={fx},gradient={gradient}\n")


from LeafNN.ConvexOptimizer.ScalarLineSearcher import ArmijoWolfeLineSearcher
def testScalarNewton4ArmijoWolfe():
    # saddle point function
    # y = x^3
    #args=[0,0,0,1]
    #y= x^4
    #argsList=[0,0,0,0,1]
    # y = x^4 + 2*x^3 + 1
    #argsList = [1,0,0,2,1]
    # y = X^3 -2*x+2
    #argsList = [2,-2,0,1]
    # y = x^2-5x+6
    #argsList = [6,-5,1]
    # y = x^2
    argsList = [0,0,1.0/1e10]
    LS = ArmijoWolfeLineSearcher(calPolyF,calPolyFAndGrad)
    LS.setSigma(0.5,0.4)
    #LS.setSigma(1.0)
    #LS = None
    sn = SN(calPolyF,calPolyFAndGrad)
    initX = 10.0
    (x,fx,gradient) = sn.calRoot(initX,argsList,lineSearcher=LS)
    Log.Debug("testBaseScalarNewton",f"x={x},fx={fx},gradient={gradient}\n")


#testBaseScalarNewton1()
#testBaseScalarNewton2()
#testScalarNewton1()
#testBaseScalarNewton3()
#testScalarNewton2()
#testScalarNewton3Sadd()
#testScalarNewton4Armijo()
testScalarNewton4ArmijoWolfe()
