import math
from LeafNN.utils.Log import Log

MsgTagBaseLS= "BaseLineSearcher"
class BaseLineSearcher:
    def __init__(self,calFFunc,calFAndGradFunc,alphaScaleMulti=0.5,maxIteraion = 20):
        self.calFFunc = calFFunc
        self.calFAndGradFunc = calFAndGradFunc
        self.maxIteraion = maxIteraion
        self.epslion=1e-8
        self.alhpaSC = alphaScaleMulti
    
    def lineSearch(self,X,d,f1,grad1,*funcArgs):
        alpha = 1.0
        iterNum = 0
        #(f1,grad1) = self.calFAndGradFunc(X,*funcArgs)
        #Log.Debug(MsgTagArmijo,f"fisrt X={X},d={d}")
        while iterNum < self.maxIteraion:
            f2 = self.calFFunc(X + alpha*d,*funcArgs)
            lhs = abs(f2)
            rhs =  abs(f1)#(1-self.sigma*alpha)*abs(f1)
            #rhs = f1 + self.sigma*alpha*grad1*d 
            if lhs<=rhs:
                Log.Info(MsgTagBaseLS,f"f2={f2},f1={f1},rhs={rhs},alpha={alpha},LineiterNum={iterNum},d={d}")
                return alpha
            else:
                alpha =self.alhpaSC*alpha
                if alpha < self.epslion:
                    break
            iterNum+=1
        Log.Warning(MsgTagBaseLS,f"failed to find the alpha,alpha={alpha},iterNum={iterNum}")
        return alpha

MsgTagArmijo = "ArmijoLineSearcher"
class ArmijoLineSearcher(BaseLineSearcher):
    def __init__(self,calFFunc,calFAndGradFunc,alphaScaleMulti=0.5,maxIteraion = 20,sigma = 0.4):
        super().__init__(calFFunc,calFAndGradFunc,maxIteraion)
        self.sigma = sigma
        self.alphaSC = alphaScaleMulti
    
    def setSigma(self,sigmaValue):
        self.sigma = sigmaValue

    def lineSearch(self,X,d,f1,grad1,*funcArgs):
        alpha = 1.0
        iterNum = 0
        #(f1,grad1) = self.calFAndGradFunc(X,*funcArgs)
        #Log.Debug(MsgTagArmijo,f"fisrt X={X},d={d}")
        while iterNum < self.maxIteraion:
            f2 = self.calFFunc(X + alpha*d,*funcArgs)
            lhs = abs(f2)
            # f2<=f1+sigma*alpha*grad.T*dk
            rhs = f1+self.sigma*alpha*d*grad1 # (1-self.sigma*alpha)*abs(f1)
            #rhs = f1 + self.sigma*alpha*grad1*d 
            if lhs<=abs(rhs):
                Log.Info(MsgTagArmijo,f"f2={f2},f1={f1},rhs={rhs},alpha={alpha},LineiterNum={iterNum},d={d}")
                return alpha
            else:
                alpha =self.alphaSC*alpha
                if alpha < self.epslion:
                    break
            iterNum+=1
        Log.Warning(MsgTagArmijo,f"failed to find the alpha,alpha={alpha},iterNum={iterNum}")
        return alpha


MsgTagArmijo = "ArmijoWolfeLineSearcher"
class ArmijoWolfeLineSearcher(BaseLineSearcher):
    def __init__(self,calFFunc,calFAndGradFunc,alphaScaleMulti=0.5,maxIteraion = 20,sigma1 = 0.4,sigma2=0.5):
        super().__init__(calFFunc,calFAndGradFunc,maxIteraion)
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.alphaSC = alphaScaleMulti
    
    def setSigma(self,sigma1,sigma2):
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def lineSearch(self,X,d,f1,grad1,*funcArgs):
        alpha = 1.0
        iterNum = 0
        #(f1,grad1) = self.calFAndGradFunc(X,*funcArgs)
        #Log.Debug(MsgTagArmijo,f"fisrt X={X},d={d}")
        while iterNum < self.maxIteraion:
            (f2,grad2) = self.calFAndGradFunc(X + alpha*d,*funcArgs)
            # f2<=f1+sigma*alpha*f'(x1)*dk
            lhs = f2
            rhs = f1+self.sigma1*alpha*d*grad1 # (1-self.sigma*alpha)*abs(f1)
            armijoFit =  abs(lhs)<=abs(rhs)
            #rhs = f1 + self.sigma*alpha*grad1*d 
            # wolfe condition2: -d*f'(x1+alpha*d)>-d*f'(x1)*sigma2
            # strong wolfe condition2: abs(d*f'(x1+alpha*d))>abs(d*f'(x1)*sigma2)
            #wolfeFit = grad2*d<grad1*self.sigma2*d# abs(grad2)>abs(grad1*self.sigma2)
            wolfeFit = abs(grad2)>abs(grad1*self.sigma2)
            if armijoFit:#armijo fit
                Log.Info(MsgTagArmijo,f"armijoFit f2={f2},f1={f1},rhs={rhs},alpha={alpha},LineiterNum={iterNum},d={d}")
            if wolfeFit:
                    Log.Info(MsgTagArmijo,f"wolfe Fit f2={f2},f1={f1},grad1={grad1},grad2={grad2},LineiterNum={iterNum},alpha={alpha},d={d}")
            if armijoFit and wolfeFit:
                return alpha
            else:
                alpha =self.alphaSC*alpha
                if alpha < self.epslion:
                    break
            iterNum+=1
        Log.Warning(MsgTagArmijo,f"failed to find the alpha,alpha={alpha},iterNum={iterNum}")
        return alpha








