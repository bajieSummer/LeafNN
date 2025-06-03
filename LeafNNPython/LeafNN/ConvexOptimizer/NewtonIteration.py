import math
from LeafNN.utils.Log import Log
from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.Bases.MatrixLinear import MatrixLinear as ML
from .LineSearcher import ArmijoWolfeLineSearcher
from .NewtonMinST import NewtonMinST
NewtonMsgTag = "NewtonIteration"

class NewtonIteration:
    def __init__(self,calFFunc,calFAndGradient,maxIteration=200,epslion =1e-20,skipGrad0Eps=1e-5):
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
        self.skipGrad0Eps = skipGrad0Eps
        #self.defaultLineSh = ArmijoWolfeLineSearcher(calFFunc,calFAndGradient)
        self.hessianDamp = self.epslion*self.epslion
        self.lamadaK = 1.0 -skipGrad0Eps
        
    def calD(gradient,gradientSquare,fx):
        #d*gradient = -fx
        # plan1 similar like 2d,
        #d = -1.0*fx/gradient
        # plan2:
        #d =-1.0*fx/math.sqrt(abs(fx))*gradient/gradientSqrt
        # plan3
        #d =-1.0*fx/(abs(fx))*gradient/math.sqrt(gradientSquare)
        # plan4
        #gradientSqrt = math.sqrt(gradientSquare)
       # d = -1.0*fx*gradient/gradientSqrt
        # 1. the gradient is the normal vector of F=f(x,y)-z =0   (df/dx,df/dy,-1)
        # the tagent plane is _|_  the normal vector
        # 
        t = -fx/gradientSquare
        d = t*gradient
        return d
    
    def calRoot(self,initX,*FuncGradArgs,customLineSearcher=None):
        """
        calRoot of f,
        intX->first search point
        funcGradArgs-> parameters of X
        customLineSearcher-> if is None: then defaultLineSearcher = ArmijoWolfeLineSearcher
        return (X,fx,gradient)
        """
        lineSh = customLineSearcher
        if lineSh is None:
            lineSh = self.defaultLineSh
        iterNum = 0
        X = initX
        fx = None
        gradient = None
        while iterNum < self.maxIteration:
            (fx,gradient) = self.calFuncAndGradient(X,*FuncGradArgs)
            # f'(xk)(xk+1 - xk) + f(xk) = 0
            # xk+1 = xk - f(xk)/f'(xk)
            # d = -f(xk)/f'(xk)
            if math.isclose(fx,0.0,abs_tol=self.epslion):
                Log.Info(NewtonMsgTag,f"found result root={X},iterNum={iterNum}\n")
                return (X,fx,gradient)
            
            gradientSquare= gradient.T@gradient
            if math.isclose(gradientSquare,0.0,abs_tol=self.epslion):
                X = X + self.skipGrad0Eps# self.epslion
                iterNum+=1
                continue
            
            #d = -fx/gradient
            d = NewtonIteration.calD(gradient,gradientSquare,fx)
            Log.Debug(NewtonMsgTag,f"iterNum={iterNum},X={X},fx={fx},d={d},grad={gradient},d={d}")
            alpha = lineSh.lineSearch(X,d,fx,gradient,*FuncGradArgs)
            lambda_k =1.0# min(1, 0.5/abs(fx))
            X = X +d*(lambda_k*alpha)
            iterNum+=1
        Log.Warning(NewtonMsgTag,f"Reached Maximum iterations X={X},NotFoundRoots,f={fx},f'={gradient}\n")
        return (X,fx,gradient)

    def calMin(self,initX,calHessianFunc,*FuncGradArgs,customLineSearcher=None,histDataCollector=None):
        """
        calMin of f,
        intX->first search point
        calHessianFunc->cal f''(X)
        funcGradArgs-> parameters of for calF,calFAndGrad,calHessianFunc
        customLineSearcher-> if is None: then defaultLineSearcher = ArmijoWolfeLineSearcher
        return (X,fx,gradient,f'')
        """
        funcTuple = (self.calFFunc,self.calFuncAndGradient,calHessianFunc)
        newtonArgsTuple = (self.maxIteration,self.epslion,self.skipGrad0Eps,self.hessianDamp)
        return NewtonMinST.calMin(initX,funcTuple,newtonArgsTuple,*FuncGradArgs,customLineSearcher=customLineSearcher,histDataCollector=histDataCollector)

    

       


