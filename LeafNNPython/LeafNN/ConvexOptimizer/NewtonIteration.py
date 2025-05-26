import math
from LeafNN.utils.Log import Log
from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.Bases.MatrixLinear import MatrixLinear as ML
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
        self.hessianDamp = self.epslion*self.epslion
        
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

    def calMinD(self,gradient,gradientSquare,fx,HessMatrix,detH):
        #if detH ==0:
        #t = -abs(fx)/gradientSquare
        #d = t*gradient
        # if math.isclose(detH,0.0,abs_tol=self.epslion):
        #     #d = -1.0*gradient
        #     if math.isclose(fx,0.0,abs_tol=self.skipGrad0Eps): # avoid when f(x)=0, but not the minimum
        #         d =-1.0*gradient #-1.0/gradientSquare*gradient
        #         Log.Debug(NewtonMsgTag,f"fx_close to 0,fx={detH},d=\n{d},gradient={gradient}\n")
        #     else:
        #         d = -abs(fx)/gradientSquare*gradient # more stable
        #     Log.Debug(NewtonMsgTag,f"detH close to 0,detH={detH},d=\n{d},HessMatrix={HessMatrix}")
        # else:
        H_d = HessMatrix + MM.identity(HessMatrix.shape[0])*self.hessianDamp
        d = abs(ML.getInverse(H_d))@gradient*(-1.0)
        return d

    def calMin(self,initX,calHessianFunc,*FuncGradArgs,customLineSearcher=None,histDataCollector=None):
        """
        calMin of f,
        intX->first search point
        calHessianFunc->cal f''(X)
        funcGradArgs-> parameters of for calF,calFAndGrad,calHessianFunc
        customLineSearcher-> if is None: then defaultLineSearcher = ArmijoWolfeLineSearcher
        return (X,fx,gradient,f'')
        """
        lineSh = customLineSearcher
        if lineSh is None:
            lineSh = self.defaultLineSh
        iterNum = 0
        X = initX
        fx = None
        N = len(X)
        lastd = None
        gradient = None
        hessM = None
        while iterNum < self.maxIteration:
            (fx,gradient) = self.calFuncAndGradient(X,*FuncGradArgs)
            if histDataCollector is not None:
                histDataCollector.append((X,fx,gradient))
            # if math.isclose(fx,0.0,abs_tol=self.epslion):
            #     Log.Info(NewtonMsgTag,f"found result root={X},iterNum={iterNum}\n")
            #     return (X,fx,gradient)
            
            gradientSquare= gradient.T@gradient
            hessM = calHessianFunc(X,*FuncGradArgs)
            detH = ML.det(hessM)
             # avoid special saddle point 
            # if(detH==0):
            #     X = X + self.skipGrad0Eps# self.epslion
            #     iterNum+=1
            #     Log.Debug(NewtonMsgTag,f"skip for detH=0-> iterNum={iterNum},X={X},fx={fx},grad=\n{gradient}\n,HesssianMatrix=\n{hessM}")
            #     continue
           
            if math.isclose(gradientSquare,0.0,abs_tol=self.epslion*self.epslion):
                # avoid saddle point and some other flat areas
                # detH <0 and grad ==0,   # normal saddle point z=x^2-y^2
                # detH~0, and grad = 0, might flat area or degenerated saddle point z=x^4-y^4
                Log.Debug(NewtonMsgTag,f"try find the min-> iterNum={iterNum},detH={detH},X={X},fx={fx},grad=\n{gradient}\n,HesssianMatrix=\n{hessM}\n,lastd={lastd}\n")
                if(detH<=self.skipGrad0Eps):#self.epslion):# skipGrad0Eps
                    Log.Debug(NewtonMsgTag,f"saddle or other situation")
                    if lastd is None:
                         X = X + self.skipGrad0Eps
                    else:
                        dt=lastd*1.0/(math.sqrt(lastd.T@lastd))
                        X = X + self.skipGrad0Eps*dt# self.epslion
                    iterNum+=1
                    continue
                Log.Debug(NewtonMsgTag,f"try find the min-> succeed-min")
                return (X,fx,gradient)
            #d = -fx/gradient
            d = self.calMinD(gradient,gradientSquare,fx,hessM,detH)
            Log.Debug(NewtonMsgTag,f"iterNum={iterNum},X={X},fx={fx},d={d},grad=\n{gradient}\n,HesssianMatrix=\n{hessM}")
            alpha = lineSh.lineSearchMin(X,d,fx,gradient,*FuncGradArgs)
            lambda_k =1.0# min(1, 0.5/abs(fx))
            X = X +d*(lambda_k*alpha)
            iterNum+=1
            lastd = d
           
        Log.Warning(NewtonMsgTag,f"Reached Maximum iterations X={X},NotFoundRoots,f={fx},f'={gradient}\n")
        return (X,fx,gradient)

    

       


