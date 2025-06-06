import demoInit 
from LeafNN.utils.Log import Log
from HelperUtils.Helper_PolyFunc import PolyFuncHelper 
from LeafNN.ConvexOptimizer.NewtonIteration import NewtonIteration
from HelperUtils.Helper_PlotNewtonIteration import PlotNewtonHelper
#from LeafNN.Bases.MathMatrix import MathMatrix as MM
tag_msg="TestNewtonBFGS"

class CaseArgs:
    def __init__(self,drawPlots=False,maxIteration=200,tflatMax=100,epslion=1e-20,customLineSearcher=None):
        self.drawPlots = drawPlots
        self.maxIteration = maxIteration
        #self.skipGrad0Eps = skipGrad0Eps
        self.tflatMax = tflatMax
        self.epslion =epslion
        self.customLineSearcher = customLineSearcher

def runCase(caseResults,f_str,initXArr,argsList,funcTuple,caseArgs:CaseArgs,expectF):
    calF,calFAndGrad = funcTuple
    newton = NewtonIteration(calF,calFAndGrad)
    newton.maxIteration = caseArgs.maxIteration
    newton.tflatMax = caseArgs.tflatMax
    newton.epslion = caseArgs.epslion
    #N variables
    N = len(initXArr)
    #initX=MM.array(initXArr).reshape(N,1)
    LS = caseArgs.customLineSearcher
    hisData = []
    (X,fx,grad)=newton.calMinBFGS(initXArr,argsList,customLineSearcher=LS,histDataCollector=hisData)
    iterNums = len(hisData)
    if caseArgs.drawPlots: 
        PlotNewtonHelper.drawHisData(hisData,0,calF,argsList,f_str)
    Log.Debug(tag_msg, f"case_f={f_str},X=\n{X}\n,fx={fx},grad=\n{grad}\n")
    caseResults.append((f_str,X,fx,grad,iterNums,initXArr,expectF))
   # return (X,fx,grad,iterNums)

# def runLogCase(casesResults,f_str,initXArr,funcArgList,caseArgs,expectF):#funcTuple
#     funcTuple = (logFuncLinear,logFuncLinearGrad,logLinearHessian)
#     #caseArgs = CaseArgs(False,400,1e-5,1e-20)
#     N = len(initXArr)
#     [wArr,b,u,c,base]=funcArgList
#     W = MM.array(wArr).reshape([N,1])
#     argsList=[[W,b],[u,c,base]]
#     (X,fx,grad,iterNums)=runCase(f_str,initXArr,argsList,funcTuple,caseArgs,expectF)

def printcaseResults(caseResults):
    for case in caseResults:
        (f_str,X,fx,grad,iterNums,initXArr,expectLessF) = case
        isPasStr = "caseSucceed"
        if fx >expectLessF:
            isPasStr = "caseFailed"
        Log.Debug(tag_msg, f"isPasStr={isPasStr} case_f={f_str}:\n initX={initXArr}\n,lastfx={fx},expectLessF={expectLessF},iterNum={iterNums}\n,X=\n{X}\n,grad=\n{grad}\n")


def testSimpleNewtonBFGS():
    casesRes =[]
    casesArgs =CaseArgs(True,200,100)
    funcTuple=(PolyFuncHelper.calDPolyF,PolyFuncHelper.calDPolyFAndGrad)

    # f_str="x^2"
    # argsList=[[0,0,1]]
    # initXArr = [100]
    # expectLessF = 0.0
    # runCase(casesRes,f_str,initXArr,argsList,funcTuple,casesArgs,expectLessF)

    # f_str="x^8"
    # argsList=[[0,0,0,0,0,0,0,0,1]]
    # initXArr = [100]
    # expectLessF = 0.00001
    # runCase(casesRes,f_str,initXArr,argsList,funcTuple,casesArgs,expectLessF)

    # # ##will be oscillated durning near (-0.5,1.8)
    # f_str = " x^3-2*x +2"
    # argsList=[[2,-2,0,1]]
    # initXArr=[1.2]
    # expectLessF = 0.95
    # runCase(casesRes,f_str,initXArr,argsList,funcTuple,casesArgs,expectLessF)

    # f_str = "x^3 +1" # degenerated saddle point1 localMinimal
    # argsList=[[1,0,0,1]]
    # initXArr=[0.00001] # 0.01
    # expectLessF = -1e36#-4.0e16# ,itersNum = 50
    # runCase(casesRes,f_str,initXArr,argsList,funcTuple,casesArgs,expectLessF)

   
    # f_str = "x^2 -y^2" # saddle point1 localMinimal
    # argsList=[[0,0,1],[0,0,-1]]
    # initXArr=[1,1]
    # expectLessF = -6.0e119# ,itersNum = 200
    # runCase(casesRes,f_str,initXArr,argsList,funcTuple,casesArgs,expectLessF)


    # f_str = "x^7 +1"  #saddle point 2
    # argsList=[[1,0,0,0,0,0,0,1]]
    # initXArr=[0.5]
    # expectLessF = -4.0e20# ,itersNum = 200
    # runCase(casesRes,f_str,initXArr,argsList,funcTuple,casesArgs,expectLessF)

    # f_str = "x^3 + y^3 +2"
    # argsList=[[1,0,0,1],[1,0,0,1]]
    # initXArr=[600,10]
    # expectLessF = -2.0e66# ,itersNum = 
    # runCase(casesRes,f_str,initXArr,argsList,funcTuple,casesArgs,expectLessF)

    # f_str = "x^7 + y^7+1" # saddle and flat area
    # argsList=[[1,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1]]
    # initXArr=[10,1]
    # expectLessF = -4.0e58# ,itersNum = 300
    # runCase(casesRes,f_str,initXArr,argsList,funcTuple,casesArgs,expectLessF)
  
    # f_str = "x^8" # flat area
    # argsList=[[0,0,0,0,0,0,0,0,1]]
    # initXArr=[0.1]
    # expectLessF = 1e-15# ,itersNum = 200
    # runCase(casesRes,f_str,initXArr,argsList,funcTuple,casesArgs,expectLessF)

    # f_str = "x^8 + y^8" # flat area
    # argsList=[[0,0,0,0,0,0,0,0,10000],[0,0,0,0,0,0,0,0,1]]
    # initXArr=[0.1,0.2]
    # expectLessF = 1e-15# ,itersNum = 200
    # runCase(casesRes,f_str,initXArr,argsList,funcTuple,casesArgs,expectLessF)
 
    f_str = "10*x^8 + y^8 +20" # flat area
    argsList=[[20,0,0,0,0,0,0,0,10],[0,0,0,0,0,0,0,0,1]]
    initXArr=[0.2,0.1]
    expectLessF = 20.01# ,itersNum = 200
    runCase(casesRes,f_str,initXArr,argsList,funcTuple,casesArgs,expectLessF)

    # f_str = "x^2 -y^2" # saddle point1 localMinimal
    # argsList=[[0,0,1],[0,0,-1]]
    # initXArr=[1,1]
    # expectLessF = 1e-15
    # runCase(casesRes,f_str,initXArr,argsList,funcTuple,casesArgs,expectLessF)


    printcaseResults(casesRes)

testSimpleNewtonBFGS()
    