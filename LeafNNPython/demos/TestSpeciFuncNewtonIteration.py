import demoInit
import math
from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.utils.Log import Log
from LeafNN.ConvexOptimizer.NewtonIteration import NewtonIteration
from LeafNN.ConvexOptimizer.LineSearcher import ArmijoLineSearcher
from LeafNN.ConvexOptimizer.LineSearcher import BaseLineSearcher
from LeafNN.ConvexOptimizer.LineSearcher import ZeroLineSearcher
from demos.HelperUtils.Helper_PlotNewtonIteration import PlotNewtonHelper
tag_msg ="TestSpecialNT"
class CaseArgs:
    def __init__(self,drawPlots=False,maxIteration=200,skipGrad0Eps=1e-5,epslion=1e-20,customLineSearcher=None):
        self.drawPlots = drawPlots
        self.maxIteration = maxIteration
        self.skipGrad0Eps = skipGrad0Eps
        self.epslion =epslion
        self.customLineSearcher = customLineSearcher

def logFuncLinear(X,argsList):
    """
    X will be vector [[x0],[x1],[x2]....] N*1
    [[u,c],[w,b]]
    u = argsList[0][0] :scalar
    c = argsList[0][1] :scalar
    w = argsList[1][0] :will be vector [[w0],[w1],[w2],..] N*1
    b = argsList[1][1] :scalar
  
    f = u*log(W*X+b)+c :scalar
    """
    W = argsList[0][0]
    b = argsList[0][1]
    u = argsList[1][0]
    c = argsList[1][1]
    base = argsList[1][2]
    Xl = W.T@X +b
    # check domain 
    f = None
    if Xl==0.0: # inf
        Log.Info(tag_msg,f"logFuncLinear->Xl={Xl}=0,the f will be -inf")
        f =  -1.0*u*math.inf
    elif Xl<0.0:
        Log.Error(tag_msg,f"logFuncLinear->invalid domain, Xl={Xl},which should >0,W={W},X={X},b={b}")
    else:
        f = u*math.log(Xl,base)+c
    return f

def logFuncLinearGrad(X,argsList):
    """
    
    X will be vector [[x0],[x1],[x2]....] N*1
    [[u,c,base],[w,b]]
    u = argsList[0][0] :scalar
    c = argsList[0][1] :scalar
    base =argsList[0][2] :scalar
    w = argsList[1][0] :will be vector [[w0],[w1],[w2],..] N*1
    b = argsList[1][1] :scalar
  
    f = u*log(W*X+b)+c :scalar
    f'= u/[(W*X+b)*ln(base)]*W 
    """
    W = argsList[0][0]
    b = argsList[0][1]
    u = argsList[1][0]
    c = argsList[1][1]
    base = argsList[1][2]
    Xl = W.T@X +b
    # check domain 
    f,grad = None,None
    if Xl==0.0: # inf
        Log.Info(tag_msg,f"logFuncLinearGrad->Xl={Xl}=0,the f will be -inf")
        f =  -1.0*u*math.inf
        grad = MM.zeros(W.shape)
    elif Xl<0.0:
        Log.Error(tag_msg,f"logFuncLinearGrad->invalid domain, Xl={Xl},which should >0,W={W},X={X},b={b}")
    else:
        grad = u/(Xl*math.log(base))*W
        f = u*math.log(Xl,base)+c
    return (f,grad)

def logLinearHessian(X,argsList):
    """
    X will be vector [[x0],[x1],[x2]....] N*1
    [[u,c,base],[w,b]]
    u = argsList[0][0] :scalar
    c = argsList[0][1] :scalar
    base =argsList[0][2] :scalar
    w = argsList[1][0] :will be vector [[w0],[w1],[w2],..] N*1
    b = argsList[1][1] :scalar
  
    f = u*log(W*X+b)+c :scalar
    f'= u/[(W*X+b)*ln(base)]*W 
    df''/(dxidxj) = u*wi*(WX+b)^-2*wj/ln(base)
    """
    W = argsList[0][0]
    b = argsList[0][1]
    u = argsList[1][0]
    c = argsList[1][1]
    base = argsList[1][2]
    Xl = W.T@X +b
    N = len(W)
    HessianM = MM.zeros([N,N])
    for i in range(N):
        for j in range(N):
            HessianM[i,j] = -1.0*u*W[i,0]*W[j,0]/(Xl*Xl*math.log(base))
    return HessianM

def testCaselogLinear():
    #  f = u*log_base(W*X+b)+c
    # f_str = "y = 2*ln(x)"
    # initXArr = [5]
    # wArr = [1.0]
    # b = 0.0
    # u,c,base=2.0,0.0,math.e

    f_str = "y = 2*log10(3.0x+4)-20"
    initXArr = [5]
    wArr = [3.0]
    b = 4.0
    u,c,base=2.0,-20.0,10.0

    N = len(initXArr)
    W = MM.array(wArr).reshape([N,1])
    argsList=[[W,b],[u,c,base]]
    initX = MM.array(initXArr).reshape([N,1])
    f = logFuncLinear(initX,argsList)
    f1,gradf = logFuncLinearGrad(initX,argsList)
    hessianM = logLinearHessian(initX,argsList)
    checkf = u*math.log(W[0,0]*initX[0,0]+b,base)+c
    
    Log.Debug(tag_msg,f"f_str={f_str}:\n X={initX},f={f},f1={f1},gradf={gradf},hessianM={hessianM}")
    Log.Debug(tag_msg,f"check_f={checkf},f={f} error ={checkf-f,checkf-f1}")
    




def runCase(f_str,initXArr,argsList,funcTuple,caseArgs:CaseArgs):
    calF,calFAndGrad,calHessM = funcTuple
    newton = NewtonIteration(calF,calFAndGrad)
    newton.maxIteration = caseArgs.maxIteration
    newton.skipGrad0Eps = caseArgs.skipGrad0Eps
    newton.epslion = caseArgs.epslion

    #N variables
    N = len(initXArr)
    initX=MM.array(initXArr).reshape(N,1)
    #LS = ZeroLineSearcher(calDPolyF,calDPolyFAndGrad)
    
    #LS = ArmijoLineSearcher(calDPolyF,calDPolyFAndGrad)
    #LS = BaseLineSearcher(calDPolyF,calDPolyFAndGrad)
    LS = caseArgs.customLineSearcher
   
    hisData = []
    (X,fx,grad)=newton.calMin(initX,calHessM,argsList,customLineSearcher=LS,histDataCollector=hisData)
    iterNums = len(hisData)
    if caseArgs.drawPlots: 
        PlotNewtonHelper.drawHisData(hisData,0,calF,argsList)
    Log.Debug(tag_msg, f"case_f={f_str},X=\n{X}\n,fx={fx},grad=\n{grad}\n")
    return (X,fx,grad,iterNums)

MM.set_printoptions(20,suppress=False)
def testLogLinearNewton():
    funcTuple = (logFuncLinear,logFuncLinearGrad,logLinearHessian)
    caseArgs = CaseArgs(True,200,1e-5,1e-20)
    # f_str = "y = 2*log10(3.0x+4)-20"
    # initXArr = [5]
    # wArr = [3.0]
    # b = 4.0
    # u,c,base=2.0,-20.0,10.0

    # f_str = "y = 20*ln(1000x+4)-30" # lamadaK = 0.99999
    # wArr = [1000]
    # b = 4.0
    # u,c,base=20.0,-30.0,math.e
    # initXArr = [100]


    # f_str = "y = -20*ln(1000x+4)-30" # lamadaK = 0.99999
    # wArr = [1000]
    # b = 4.0
    # u,c,base=-20.0,-30.0,math.e
    # initXArr = [100]

def runLogCase(casesResults,f_str,initXArr,funcArgList,caseArgs,expectF):#funcTuple
    funcTuple = (logFuncLinear,logFuncLinearGrad,logLinearHessian)
    #caseArgs = CaseArgs(False,400,1e-5,1e-20)
    N = len(initXArr)
    [wArr,b,u,c,base]=funcArgList
    W = MM.array(wArr).reshape([N,1])
    argsList=[[W,b],[u,c,base]]
    (X,fx,grad,iterNums)=runCase(f_str,initXArr,argsList,funcTuple,caseArgs)
    casesResults.append((f_str,X,fx,grad,iterNums,initXArr,expectF))

def printcaseResults(caseResults):
    for case in caseResults:
        (f_str,X,fx,grad,iterNums,initXArr,expectLessF) = case
        isPasStr = "caseSucceed"
        if fx >expectLessF:
            isPasStr = "caseFailed"
        Log.Debug(tag_msg, f"isPasStr={isPasStr} case_f={f_str}:\n initX={initXArr}\n,lastfx={fx},expectLessF={expectLessF},iterNum={iterNums}\n,X=\n{X}\n,grad=\n{grad}\n")




def testLogLinearNewton():
    Log.Debug(tag_msg,f"runLogLinearTestCases-->begin")
    caseResults=[]
    caseArgs = CaseArgs(False,200,1e-5,1e-20)

    f_str = "y = 2*log10(3.0x+4)-20"
    initXArr = [5]
    wArr = [3.0]
    b = 4.0
    u,c,base=2.0,-20.0,10.0
    expectf=1e-15
    runLogCase(caseResults,f_str,initXArr,[wArr,b,u,c,base],caseArgs,expectf)

    f_str = "y = 20*ln(1000x+4)-30" # lamadaK = 0.99999
    wArr = [1000]
    b = 4.0
    u,c,base=20.0,-30.0,math.e
    initXArr = [100] # 0.0
    expectf=1e-15
    runLogCase(caseResults,f_str,initXArr,[wArr,b,u,c,base],caseArgs,expectf)

    f_str = "y = -20*ln(1000x+4)-30" # lamadaK = 0.99999
    wArr = [1000]
    b = 4.0
    u,c,base=-20.0,-30.0,math.e
    initXArr = [100] # -1100
    expectf=-1000 #-1100
    runLogCase(caseResults,f_str,initXArr,[wArr,b,u,c,base],caseArgs,expectf)

    f_str = "y = -20*ln(1000x+y+4)-30" # 
    wArr = [1000,1]
    b = 4.0
    u,c,base=-20.0,-30.0,math.e
    initXArr = [1,100]
    expectf=-500
    runLogCase(caseResults,f_str,initXArr,[wArr,b,u,c,base],caseArgs,expectf)


    f_str = "y = -20*ln(1000x1+x2+100x3+4)-30" # sigular #expectf = -1154
    wArr = [1000,1,100]
    b = 4.0
    u,c,base=-20.0,-30.0,math.e
    initXArr = [0.0,100.0,10.0]
    expectf=-500
    runLogCase(caseResults,f_str,initXArr,[wArr,b,u,c,base],caseArgs,expectf)

    Log.Debug(tag_msg,f"AllLogLinearTestCasesResultsAs below casesNum={len(caseResults)}\n")
    printcaseResults(caseResults)
    Log.Debug(tag_msg,f"runLogLinearTestCases-->End") 




#testCaselogLinear()
testLogLinearNewton()

import numpy as np
from scipy.optimize import minimize
def testSpy1():
    def objective(x):
        return -20 * np.log(1000 * x + 4) - 30

    def gradient(x):
        return -20000 / (1000 * x + 4)

    def hessian(x):
        return 20_000_000 / (1000 * x + 4)**2

    # 初始猜测值（需满足 1000x + 4 > 0）
    x0 = 0.1  

    # 调用Newton-CG方法
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=gradient,
        hess=hessian,
        options={'xtol': 1e-8}
    )

    print("最优解 x =", result.x)
    print("函数最小值 f(x) =", result.fun)


#testSpy1()



    

 

    

    