from LeafNN.utils.Log import Log
import math
from .ScalarLineSearcher import BaseLineSearcher
NewtonMsgTag = "ScalarNewtonIteration"
from LeafNN.utils.Log import Log
import math
BaseNewtonMsgTag = "BaseScalarNewtonIteration"

class BaseScalarNewtonIteration:
    def __init__(self,calFAndGradient,maxIteration=200,epslion = 10e-15):
        Log.Info(BaseNewtonMsgTag,"createBaseNewtonIteration")
        if calFAndGradient is None:
            Log.Error(BaseNewtonMsgTag,"Invalid Parameter, the calFuncAndGradient is None")
            return None
        self.calFuncAndGradient = calFAndGradient
        self.maxIteration = maxIteration
        self.epslion = epslion

    def calRoot(self,initX,*FuncGradArgs):
        iterNum = 0
        X = initX
        fx = None
        gradient = None
        while iterNum < self.maxIteration:
            (fx,gradient) = self.calFuncAndGradient(X,*FuncGradArgs)
            # f'(xk)(xk+1 - xk) + f(xk) = 0
            # xk+1 = xk - f(xk)/f'(xk)
            # d = -f(xk)/f'(xk)
            if math.isclose(gradient,0.0,abs_tol=self.epslion):
                X = X + self.epslion
                iterNum+=1
                continue
            if math.isclose(fx,0.0,abs_tol=self.epslion):
                Log.Info(BaseNewtonMsgTag,f"found result root={X},iterNum={iterNum}\n")
                return (X,fx,gradient)
            Log.Debug(BaseNewtonMsgTag,f"iterNum={iterNum},X={X},fx={fx},gradient={gradient}")
            d = -fx/gradient
            X = X +d
            iterNum+=1
           
        Log.Warning(BaseNewtonMsgTag,f"Reached Maximum iterations X={X},NotFoundRoots,f={fx},f'={gradient}\n")
        return (X,fx,gradient)
    
class ScalarNewtonIteration(BaseScalarNewtonIteration):
    def __init__(self,calFFunc,calFAndGradient,maxIteration=200,epslion =1e-15,skipGrad0Eps=1e-3):
        Log.Info(NewtonMsgTag,"createScalarNewtonIteration")
        if calFFunc is None:
            Log.Error(NewtonMsgTag,"Invalid Parameter, the calFFunc is None")
            return None
        if calFAndGradient is None:
            Log.Error(NewtonMsgTag,"Invalid Parameter, the calFuncAndGradient is None")
            return None
        self.calFFunc = calFFunc
        super().__init__(calFAndGradient,maxIteration,epslion)
        self.skipGrad0Eps = skipGrad0Eps
        self.defaultLineSh = BaseLineSearcher(calFFunc,calFAndGradient)
        
    def calRoot(self,initX,*FuncGradArgs,lineSearcher=None):
        lineSh = lineSearcher
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
            if math.isclose(gradient,0.0,abs_tol=self.epslion):
                X = X + self.skipGrad0Eps
                Log.Info(NewtonMsgTag,f"gradient reach 0 ->{gradient},X={X}\n")
                iterNum+=1
                continue
           
            Log.Debug(NewtonMsgTag,f"iterNum={iterNum},X={X},fx={fx},gradient={gradient}")
            d = -fx/gradient
            alpha = lineSh.lineSearch(X,d,fx,gradient,*FuncGradArgs)
            lambda_k =1.0# min(1, 0.5/abs(fx))
            X = X +d*lambda_k*alpha
            iterNum+=1
        Log.Warning(NewtonMsgTag,f"Reached Maximum iterations X={X},NotFoundRoots,f={fx},f'={gradient}\n")
        return (X,fx,gradient)
            
            




       


