import demoInit
from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.utils.Log import Log
from LeafNN.ConvexOptimizer.NewtonIteration import NewtonIteration
from LeafNN.ConvexOptimizer.LineSearcher import ArmijoLineSearcher
from LeafNN.ConvexOptimizer.LineSearcher import BaseLineSearcher
from LeafNN.ConvexOptimizer.LineSearcher import ZeroLineSearcher
tag_msg = "testVectorNewtonIteration"

def calDPolyF(X,argsList):
    # xi_argus = argsList[i]
    # for xi-> fxi += argsList[i][0] +  argsList[i][1]*Xi  + argsList[i][2]*Xi*Xi + argsList[i][3]*Xi*Xi*Xi + ..
    # f = fx1+fx2+...fxN
    cacheX = []
    f = 0
    N = len(argsList) # which means N variables
    cacheX = MM.ones([N,1])
    for i in range(N):
        M = len(argsList[i])# means the ithe variable has M polys

        for j in range(M): 
            if i==0 and j>0:
                cacheX = MM.hstack([cacheX,X*cacheX[:,j-1:j]])
            arg = argsList[i][j]
            if arg!=0:
                f+=arg*cacheX[i][j]
    return f

def calDPolyFAndGrad(X,argsList):
    # N = len(argsList): which means N variables
    # xi_argus = argsList[i]
    # for xi-> fxi += argsList[i][0] +  argsList[i][1]*Xi  + argsList[i][2]*Xi*Xi + argsList[i][3]*Xi*Xi*Xi + ..
    # f = fx1+fx2+...fxN
    # grad[i] =  argsList[i][1] + argsList[i][2]*2*xi*xi + .. argsList[i][j]*j*xi^j+...
    cacheX = []
    # for xi cacheX[i] = [1,xi,xi^2,xi^3, ...]
    gradArr =[]
    f = 0
    N = len(argsList) # which means N variables
    cacheX = MM.ones([N,1])
    for i in range(N):
        M = len(argsList[i])# means the ithe variable has M polys
        gradArr.append(0)
        for j in range(M): 
            if i==0 and j>0:
                #print(f"X^j=\n{X*cacheX[:,j-1:j]}")
                cacheX = MM.hstack([cacheX,X*cacheX[:,j-1:j]])
            arg = argsList[i][j]
            #print(f"cacheX=\n{cacheX}")
            if arg!=0:f+=arg*cacheX[i][j]
            if j>0:
                if arg!=0:gradArr[i]+=arg*cacheX[i][j-1]*j
    grad = MM.array(gradArr).reshape([N,1])
    return (f,grad)
    
def testSimpleVectorNewtons():
    Log.Debug(tag_msg,"testSimpleVectorNewtons")
    #f = x^2+y^2
    argsList=[[0,0,1],[0,0,1]]
    newton = NewtonIteration(calDPolyF,calDPolyFAndGrad)
    initXArr = [100,100]
    initX=MM.array(initXArr).reshape(2,1)
    (X,fx,grad)=newton.calRoot(initX,argsList)
    Log.Debug(tag_msg,f"X=\n{X}\n,fx={fx},grad=\n{grad}\n")


def testCustomVectorNewtons():
    Log.Debug(tag_msg,"testSimpleVectorNewtons")
    #f = x^2+y^2
    #argsList=[[0,0,1],[0,0,1]]
    #f = 1000*x*x + y*y
    argsList=[[0,0,1000],[0,0,1]]
    initXArr = [10,20]
    #f = x*x*x -2*x +2
    # will be oscillated durning near (-0.5,1.8)
    #argsList=[[2,-2,0,1]]
    #initXArr=[-0.8]
    #f = x^3 +1
    #argsList=[[1,0,0,1]]
    #initXArr=[0]
    newton = NewtonIteration(calDPolyF,calDPolyFAndGrad)
    newton.maxIteration = 500
    newton.skipGrad0Eps = 1e-5
    #N variables
    N = len(initXArr)
    initX=MM.array(initXArr).reshape(N,1)
    LS = ZeroLineSearcher(calDPolyF,calDPolyFAndGrad)
    
    #LS = ArmijoLineSearcher(calDPolyF,calDPolyFAndGrad)
    #LS = BaseLineSearcher(calDPolyF,calDPolyFAndGrad)
    #LS = None
    (X,fx,grad)=newton.calRoot(initX,argsList,customLineSearcher=LS)
    Log.Debug(tag_msg,f"X=\n{X}\n,fx={fx},grad=\n{grad}\n")


def calDPolyFHessian(X,argsList):
    # N = len(argsList): which means N variables
    # xi_argus = argsList[i]
    # for xi-> fxi += argsList[i][0] +  argsList[i][1]*Xi  + argsList[i][2]*Xi*Xi + argsList[i][3]*Xi*Xi*Xi + ..
    # f = fx1+fx2+...fxN
    # grad[i] =  argsList[i][1] + argsList[i][2]*2*xi + .. argsList[i][j]*j*xi^(j-1)+...
    # grad2[i]= argsList[i][2]*2 + argsList[i][3]*3*2*xi +.. argsList[i][j]*j*(j-1)*xi^(j-2)+..
    cacheX = []
    # for xi cacheX[i] = [1,xi,xi^2,xi^3, ...]
    grad2Arr = []
    N = len(argsList) # which means N variables
    cacheX = MM.ones([N,1])
    for i in range(N):
        M = len(argsList[i])# means the ithe variable has M polys
        grad2Arr.append(0)
        for j in range(M): 
            if i==0 and j>0:
                #print(f"X^j=\n{X*cacheX[:,j-1:j]}")
                cacheX = MM.hstack([cacheX,X*cacheX[:,j-1:j]])
            arg = argsList[i][j]
            if j>1:
                if arg!=0:grad2Arr[i]+=arg*cacheX[i][j-2]*j*(j-1)
    gradHess = MM.diag(grad2Arr)
    return gradHess

def testCalMinWithNewtons():
    Log.Debug(tag_msg,"testCalMinWithNewtons")
    #f = x^2+y^2+10
    # argsList=[[0,0,1],[0,0,1]]
    # initXArr = [10,1]
    #f = 20*x^2+10x+y^2+5
    #argsList=[[5,10,20],[0,0,1]]
    #initXArr = [10,20]
    #f = 1000*x^2+ y^2
    argsList=[[0,0,100000],[0,0,1]]
    initXArr = [100,1]
    #f = x*x*x -2*x +2
    # will be oscillated durning near (-0.5,1.8)
    #argsList=[[2,-2,0,1]]
    #initXArr=[-0.4]
    #f = x^3 +1  saddle point 0
    # argsList=[[1,0,0,1]]
    # initXArr=[0]
    #f = x^7 +1  saddle point 0
    # argsList=[[1,0,0,0,0,0,0,1]]
    # initXArr=[0.5]
    #f = x^8 # flat area
    # argsList=[[0,0,0,0,0,0,0,0,1]]
    # initXArr=[10]
    #f = x^8 + y^8 # flat area
    # argsList=[[0,0,0,0,0,0,0,0,10000],[0,0,0,0,0,0,0,0,1]]
    # initXArr=[10,1]
    #f = x^7 + y^7+1 # flat area
    # argsList=[[1,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1]]
    # initXArr=[10,1]
    # f = x^3 + y^3 +2
    # argsList=[[1,0,0,1],[1,0,0,1]]
    # initXArr=[600,10]
    newton = NewtonIteration(calDPolyF,calDPolyFAndGrad)
    newton.maxIteration = 100
    newton.skipGrad0Eps = 1e-3
    #N variables
    N = len(initXArr)
    initX=MM.array(initXArr).reshape(N,1)
    #LS = ZeroLineSearcher(calDPolyF,calDPolyFAndGrad)
    
    #LS = ArmijoLineSearcher(calDPolyF,calDPolyFAndGrad)
    #LS = BaseLineSearcher(calDPolyF,calDPolyFAndGrad)
    LS = None
    (X,fx,grad)=newton.calMin(initX,calDPolyFHessian,argsList,customLineSearcher=LS)
    Log.Debug(tag_msg,f"X=\n{X}\n,fx={fx},grad=\n{grad}\n")

#testSimpleVectorNewtons()
#testCustomVectorNewtons()
testCalMinWithNewtons()