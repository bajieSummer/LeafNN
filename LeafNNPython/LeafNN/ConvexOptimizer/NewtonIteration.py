import math
from LeafNN.utils.Log import Log
from LeafNN.Bases.MathMatrix import MathMatrix as MM
from .LineSearcher import ArmijoWolfeLineSearcher
NewtonMsgTag = "NewtonIteration"
class NewtonIteration:
    def __init__(self,calFFunc,calFAndGradient,maxIteration=200,epslion =1e-15,skipGrad0Eps=1e-3):
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
        self.defaultLineSh = ArmijoWolfeLineSearcher(calFFunc,calFAndGradient)
        
    def calD(gradient,gradientSquare,fx):
        #d*gradient = -fx
        # plan1 similar like 2d,
        #d = -1.0*fx/gradient
        # plan2:
        #d =-1.0*fx/math.sqrt(abs(fx))*gradient/gradientSqrt
        # plan3
        #d =-1.0*fx/(abs(fx))*gradient/gradientSqrt
        # 1. the gradient is the normal vector of F=f(x,y)-z =0   (df/dx,df/dy,-1)
        # the tagent plane is _|_  the normal vector
        # 
        sgradient = MM.vstack([gradient,-1.0])
        N = len(sgradient)
        sv = MM.zeros([N-1,1])
        gN = sgradient[N-1]
        lastSV = -gradientSquare
        for i in range(N-1):
            gi = gradient[i]
            sv[i]=gi*gN 
            #lastSV -=gi*gi
        lastP0= fx 
        t = -fx/lastSV
        d = t*sv 
        #plan4
        #d = -1.0*fx/abs(fx)*gradient
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
            checkfx = self.calFFunc(X,*FuncGradArgs)
            Log.Debug(NewtonMsgTag,f"fx={fx},checkfx={checkfx}")
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
            


       


