import demoInit
from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.utils.Log import Log
from LeafNN.ConvexOptimizer.NewtonIteration import NewtonIteration
from LeafNN.ConvexOptimizer.LineSearcher import ArmijoLineSearcher
from LeafNN.ConvexOptimizer.LineSearcher import BaseLineSearcher
from LeafNN.ConvexOptimizer.LineSearcher import ZeroLineSearcher
import matplotlib.pyplot as plt

tag_msg = "testVectorNewtonIteration"

def calDPolyF(X,argsList):
    # xi_argus = argsList[i]
    # for xi-> fxi += argsList[i][0] +  argsList[i][1]*Xi  + argsList[i][2]*Xi*Xi + argsList[i][3]*Xi*Xi*Xi + ..
    # f = fx1+fx2+...fxN
    # check shape:
  
    if(X.shape !=None and len(X.shape)!=2 and X.shape[1]!=1):
        Log.Error(tag_msg,f"shape is not right, we should have (N,1), but shapeX={X.shape}")
        return None
    cacheX = []
    f = 0.0
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
    #f = 200*x^2+10x+y^2+5
    #argsList=[[5,10,200],[0,0,1]]
    #initXArr = [10,20]
    #f = 1000*x^2+ y^2
    # argsList=[[0,0,100000],[0,0,1]]
    # initXArr = [100,1]
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
    argsList=[[1,0,0,1],[1,0,0,1]]
    initXArr=[600,10]
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

def getContourMesh(x0_min,x0_max,x1_min,x1_max,func,argsList,steps=400):
    if x0_max == x0_min: x0_max +=1
    if x1_max == x1_min: x1_max +=1
    # make the shape not scale
    x0_min =min(x0_min,x1_min)
    x1_min =x0_min
    x0_max = max(x0_max,x1_max)
    x1_max = x0_max
    fmin = None 
    fmax = None

    X0_l = MM.linspace(x0_min, x0_max, steps)
    steps1 =int(steps*(x1_max-x1_min)/(x0_max-x0_min))
    X1_l = MM.linspace(x1_min, x1_max, steps1)
    X0, X1 = MM.meshgrid(X0_l, X1_l)
    Z =  MM.zeros([steps1,steps])
    for i in range(steps1):
        for j in range(steps):
            xi = MM.array([[X0[0,j]],[X1[i,0]]])
            Z[i,j] = func(xi,argsList)
            if fmin == None: fmin ,fmax=Z[i,j],Z[i,j]
            fmin=min(fmin,Z[i,j])
            fmax=max(fmax,Z[i,j])
    return (X0,X1,Z,fmin,fmax)


def drawHisData(hisData,extend=2,calPolyF=None,argsList=None,f_str=""):
    X0data =[]
    X1data = []
    X0_min = None
    X0_max = None 
    X1_min = None 
    X1_max = None
    dimension = len(hisData[0][0])
    for d in hisData:
        X = d[0]
        f = d[1]
        grad =d[2]
        x0 = X[0,0]
        x1 = f
        if dimension>1:x1 = X[1,0]
        X0data.append(x0)
        X1data.append(x1)
        if X0_min is None:X0_min,X0_max =x0,x0
        X0_min = min(x0,X0_min)
        X0_max = max(x0,X0_max)
        if X1_min is None:X1_min,X1_max =x1,x1
        X1_min = min(x1,X1_min)
        X1_max = max(x1,X1_max)
    label ='x0_x1'

    if dimension ==1:
        label='XY'
    #draw x0x1 points and draw x0x1 line

   
    iterNum=len(X0data)
    # draw contour lines
    if dimension >1 and calPolyF is not None:
        (Xt,Yt,Zt,fmin,fmax) = getContourMesh(X0_min-extend,X0_max+extend,X1_min-extend,X1_max+extend,calPolyF,argsList)
        Log.Debug(tag_msg,f"Xt=\n{Xt}\n,Yt=\n{Yt}\n,Zt=\n{Zt}\n")
        Log.Debug(tag_msg,f"getMinMax_from CoutourMap fmin={fmin},fmax={fmax}\n")
        contour = plt.contourf(Xt, Yt, Zt, levels=MM.linspace(fmin, fmax, 30), cmap='viridis')
        ###plt.contour(Xt, Yt, Zt, levels=10, linestyles='dashed', colors='black')
        cbar = plt.colorbar(contour)
        cbar.set_label('Z Value')
    else: # draw function line
        xt = MM.linspace(X0_min-extend,X0_max+extend,num=200)
        yt = []
        for xti in xt: yt.append(calPolyF(MM.array([[xti]]),argsList))
        plt.plot(xt,yt,color='blue',linestyle="--", label=label)

    plt.scatter(X0data, X1data, color='red', s=5, marker='D', label=label)
    plt.plot(X0data,X1data,color='black',label=label)
    plt.title(f"Draw Iters:f={f_str},convergeNum={iterNum}")
    plt.show()
    

            

def testH0CasesCalMinWithNewtons():
    # # f = x^2 +y^2 
    #argsList=[[0,0,1],[0,0,1]]
    #initXArr = [100,10]

    # # f = x^2 +y 
    # argsList=[[0,0,1],[0,1]]
    # initXArr = [100,10]

    #f = x^3-2*x +2
    # will be oscillated durning near (-0.5,1.8)
    # argsList=[[2,-2,0,1]]
    # initXArr=[1.2]
    #f = x^3 +1  saddle point 0
    # argsList=[[1,0,0,1]]
    # initXArr=[0]
    #f = x^7 +1  saddle point 0
    #argsList=[[1,0,0,0,0,0,0,1]]
    #initXArr=[0.5]
    #f = x^8 # flat area
    argsList=[[0,0,0,0,0,0,0,0,1]]
    initXArr=[0.1]

    newton = NewtonIteration(calDPolyF,calDPolyFAndGrad)
    newton.maxIteration = 300
    newton.skipGrad0Eps = 1e-3
    newton.epslion = 1e-20
    #N variables
    N = len(initXArr)
    initX=MM.array(initXArr).reshape(N,1)
    #LS = ZeroLineSearcher(calDPolyF,calDPolyFAndGrad)
    
    #LS = ArmijoLineSearcher(calDPolyF,calDPolyFAndGrad)
    #LS = BaseLineSearcher(calDPolyF,calDPolyFAndGrad)
    LS = None
    hisData = []
    (X,fx,grad)=newton.calMin(initX,calDPolyFHessian,argsList,customLineSearcher=LS,histDataCollector=hisData)
    iterNums = len(hisData)
    drawHisData(hisData,10,calDPolyF,argsList)
    Log.Debug(tag_msg,f"X=\n{X}\n,fx={fx},grad=\n{grad}\n")

def runCase(initXArr,argsList,f_str="",drawPlots=False,maxIteration=200,skipGrad0Eps=1e-5,epslion=1e-20):
    newton = NewtonIteration(calDPolyF,calDPolyFAndGrad)
    newton.maxIteration = maxIteration
    newton.skipGrad0Eps = skipGrad0Eps
    newton.epslion = epslion
    #N variables
    N = len(initXArr)
    initX=MM.array(initXArr).reshape(N,1)
    #LS = ZeroLineSearcher(calDPolyF,calDPolyFAndGrad)
    
    #LS = ArmijoLineSearcher(calDPolyF,calDPolyFAndGrad)
    #LS = BaseLineSearcher(calDPolyF,calDPolyFAndGrad)
    LS = None
   
    hisData = []
    (X,fx,grad)=newton.calMin(initX,calDPolyFHessian,argsList,customLineSearcher=LS,histDataCollector=hisData)
    iterNums = len(hisData)
    if drawPlots: 
        drawHisData(hisData,3,calDPolyF,argsList)
    Log.Debug(tag_msg, f"case_f={f_str},X=\n{X}\n,fx={fx},grad=\n{grad}\n")
    return (X,fx,grad,iterNums)


def runCases():
    casesRes=[]
    f_str = "x^2 +y^2"
    argsList=[[0,0,1],[0,0,1]]
    initXArr = [100,10]
    (X,fx,grad,iterNums)=runCase(initXArr,argsList,f_str,False)
    expectLessF = 0.0
    casesRes.append((f_str,X,fx,grad,iterNums,expectLessF,initXArr))

    f_str = "1e10*x^2 +y^2"
    argsList=[[0,0,1e10],[0,0,1]]
    initXArr = [10000,1]
    (X,fx,grad,iterNums)=runCase(initXArr,argsList,f_str,False)
    expectLessF = 1e-20
    casesRes.append((f_str,X,fx,grad,iterNums,expectLessF,initXArr))

    
    f_str = "x^2 +y"
    argsList=[[0,0,1],[0,1]]
    initXArr = [100,10]
    (X,fx,grad,iterNums)=runCase(initXArr,argsList,f_str,False)
    expectLessF = -1.0e32 #200 iterations iterNum = 200
    casesRes.append((f_str,X,fx,grad,iterNums,expectLessF,initXArr))

    # ##will be oscillated durning near (-0.5,1.8)
    f_str = " x^3-2*x +2"
    argsList=[[2,-2,0,1]]
    initXArr=[1.2]
    (X,fx,grad,iterNums)=runCase(initXArr,argsList,f_str,False)
    expectLessF = 0.92# fx 0.911337892,itersNum = 5
    casesRes.append((f_str,X,fx,grad,iterNums,expectLessF,initXArr))

    f_str = "x^2 -y^2" # saddle point1 localMinimal
    argsList=[[0,0,1],[0,0,-1]]
    initXArr=[1,1]
    (X,fx,grad,iterNums)=runCase(initXArr,argsList,f_str,False)
    expectLessF = -6.0e119# ,itersNum = 200
    casesRes.append((f_str,X,fx,grad,iterNums,expectLessF,initXArr))

    f_str = "x^3 +1" # degenerated saddle point1 localMinimal
    argsList=[[1,0,0,1]]
    initXArr=[10.0]
    (X,fx,grad,iterNums)=runCase(initXArr,argsList,f_str,False,200,1e-5)
    expectLessF = -1e40#-4.0e16# ,itersNum = 200
    casesRes.append((f_str,X,fx,grad,iterNums,expectLessF,initXArr))


    f_str = "x^7 +1"  #saddle point 2
    argsList=[[1,0,0,0,0,0,0,1]]
    initXArr=[0.5]
    (X,fx,grad,iterNums)=runCase(initXArr,argsList,f_str,False)
    expectLessF = -4.0e20# ,itersNum = 200
    casesRes.append((f_str,X,fx,grad,iterNums,expectLessF,initXArr))

    f_str = "x^3 + y^3 +2"
    argsList=[[1,0,0,1],[1,0,0,1]]
    initXArr=[600,10]
    (X,fx,grad,iterNums)=runCase(initXArr,argsList,f_str,False)
    expectLessF = -2.0e66# ,itersNum = 
    casesRes.append((f_str,X,fx,grad,iterNums,expectLessF,initXArr))

    f_str = "x^7 + y^7+1" # saddle and flat area
    argsList=[[1,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1]]
    initXArr=[10,1]
    (X,fx,grad,iterNums)=runCase(initXArr,argsList,f_str,False,300,1e-5)
    expectLessF = -4.0e58# ,itersNum = 300
    casesRes.append((f_str,X,fx,grad,iterNums,expectLessF,initXArr))
    
    f_str = "x^8" # flat area
    argsList=[[0,0,0,0,0,0,0,0,1]]
    initXArr=[0.1]
    (X,fx,grad,iterNums)=runCase(initXArr,argsList,f_str,False,200,1e-5)
    expectLessF = 1e-15# ,itersNum = 200
    casesRes.append((f_str,X,fx,grad,iterNums,expectLessF,initXArr))

    f_str = "x^8 + y^8" # flat area
    argsList=[[0,0,0,0,0,0,0,0,10000],[0,0,0,0,0,0,0,0,1]]
    initXArr=[0.1,0.2]
    (X,fx,grad,iterNums)=runCase(initXArr,argsList,f_str,False,200,1e-5)
    expectLessF = 1e-15# ,itersNum = 200
    casesRes.append((f_str,X,fx,grad,iterNums,expectLessF,initXArr))
    
    f_str = "10*x^8 + y^8 +20" # flat area
    argsList=[[20,0,0,0,0,0,0,0,10],[0,0,0,0,0,0,0,0,1]]
    initXArr=[0.2,0.1]
    (X,fx,grad,iterNums)=runCase(initXArr,argsList,f_str,False,200,1e-5)
    expectLessF = 20.01# ,itersNum = 200
    casesRes.append((f_str,X,fx,grad,iterNums,expectLessF,initXArr))
    
    print("casesRunCollection Result-------->>>>\n")
    for case in casesRes:
        (f_stri,X,fx,grad,iterNums,expectLessF,initXArr) = case
        isPasStr = "caseSucceed"
        if fx >expectLessF:
            isPasStr = "caseFailed"
        Log.Debug(tag_msg, f"isPasStr={isPasStr}\n case_f={f_stri}:\n initX={initXArr}, lastfx={fx},expectF={expectLessF},iterNum={iterNums},X=\n{X}\n,grad=\n{grad}\n")



#testSimpleVectorNewtons()
#testCustomVectorNewtons()
#testCalMinWithNewtons()
#testH0CasesCalMinWithNewtons()
runCases()