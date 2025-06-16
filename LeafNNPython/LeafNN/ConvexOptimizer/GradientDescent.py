import math
from LeafNN.utils.Log import Log
from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.Bases.MatrixLinear import MatrixLinear as ML
from .LineSearcher import ArmijoWolfeLineSearcher
from .GradientDescentST import GradientDescentST

NewtonMsgTag = "NewtonIteration"

class GradientDescent:
    def __init__(self,calFFunc,calFAndGradient,maxIteration=200,epslion =1e-20,alpha=1.0):
        Log.Info(NewtonMsgTag,"createNewtonIteration")
        if calFFunc is None:
            Log.Error(NewtonMsgTag,"Invalid Parameter, the calFFunc is None")
            return None
        if calFAndGradient is None:
            Log.Error(NewtonMsgTag,"Invalid Parameter, the calFuncAndGradient is None")
            return None
        self.calFFunc = calFFunc
        self.calFuncAndGradient = calFAndGradient
        self.maxIteration = maxIteration
        self.epslion = epslion
        self.alpha = alpha
    
    def calMinGlobal(initX,funcTuple,GDArgsTuple,*FuncGradArgs,customLineSearcher=None,histDataCollector=None):
        Log.Debug(tag_msg,f"calMin")
        """
        calMin of f,
        intX->first search point
        funcTuple =(calF,calFAndGradient,calHessianMatrix)
        funcGradArgs-> parameters of for calF,calFAndGradient,calHessianMatrix
        (maxIteration,epsilon,skipGrad0,hessianDamp) ->newtonArgsTuple
        customLineSearcher-> if is None: then defaultLineSearcher = ArmijoWolfeLineSearcher
        return (X,fx,gradient,f'')
        """
        (calF,calFAndGradient) = funcTuple
        (maxIteration,epsilon,alphaRate) = GDArgsTuple
        lineSh = customLineSearcher
   
        if lineSh is None:
            lineSh = ArmijoLineSearcher(calF,calFAndGradient)
        iterNum = 0
        X = initX
        fx = None
        N = len(X)
        
        while iterNum < maxIteration:
            (fx,gradient) = calFAndGradient(X,*FuncGradArgs)
            if histDataCollector is not None:
                histDataCollector.append((X,fx,gradient))
        
            gradientSquare= gradient.T@gradient
            if(fx == -1.0*math.inf): # already to the least
                return (X,fx,gradient)
            d= -1.0*gradient
            d = alphaRate*d
            alpha = lineSh.lineSearchMin(X,d,fx,gradient,*FuncGradArgs)
            # min(1, 0.5/abs(fx))
            Log.Debug(tag_msg,f"iterNum={iterNum},X={X},fx={fx},d={d},grad=\n{gradient}\n,alphaRate=\n{alphaRate}")
            X = X +alpha*d
            Log.Debug(tag_msg,f"iterNum={iterNum}-->,newX={X} oldX={X-d*alpha} d={d},alpha={alpha}")
            iterNum+=1
            lastd = d
           
        Log.Warning(tag_msg,f"Reached Maximum iterations X={X},not found min,f={fx},f'={gradient}\n")
        return (X,fx,gradient)

    def calMin(self,initX,*FuncGradArgs,customLineSearcher=None,histDataCollector=None):
        """
        calMin of f,
        intX->first search point
        funcGradArgs-> parameters of for calF,calFAndGrad,calHessianFunc
        customLineSearcher-> if is None: then defaultLineSearcher = ArmijoWolfeLineSearcher
        return (X,fx,gradient,f'')
        """
        funcTuple = (self.calFFunc,self.calFuncAndGradient)
        GDArgsTuple = (self.maxIteration,self.epslion,self.alpha)
        return GradientDescent.calMinGlobal(initX,funcTuple,newtonArgsTuple,*FuncGradArgs,customLineSearcher=customLineSearcher,histDataCollector=histDataCollector)
   