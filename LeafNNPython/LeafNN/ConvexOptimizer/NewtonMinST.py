from LeafNN.utils.Log import Log
from LeafNN.ConvexOptimizer.LineSearcher import ArmijoWolfeLineSearcher
from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.Bases.MatrixLinear import MatrixLinear as ML
import math
tag_msg="NewtonMinST"
class NewtonMinST:
    def _calMinD(fx,gradient,HessMatrix,gradientSquare,detH,epsilon,skipGrad0Eps,hessianDamp):
        #if detH ==0:
        #t = -abs(fx)/gradientSquare
        #d = t*gradient
        isDetH0 = math.isclose(detH,0.0,abs_tol=epsilon)
        isGradient0 =  math.isclose(gradientSquare,0.0,abs_tol=epsilon*epsilon)
        isf0 = math.isclose(fx,0.0,abs_tol=skipGrad0Eps)
        if (isDetH0 and isGradient0):
            # saddle point or flat area:
            # try something new to distract from saddle point, and marching on the new point
            t = 1.0/MM.sqrt(gradientSquare)
            d = -1.0*gradient*t*0.0001
            return d
            #Log.Debug(tag_msg,f"take adventures-->t={t},d={d}")

        # if isDetH0:
        #     if not isGradient0:
        #         # means exist gradient0 on some direction but can escape from other direction
        #         if isf0:
        #             t = 1.0/MM.sqrt(gradientSquare)
        #             #t = min(1.0/MM.sqrt(gradientSquare),1e10) # accelerate
        #             #d = -1.0*t*gradient # take a try might oscillate
        #             d = -1.0*gradient*t
        #         else: 
        #             t = min(abs(fx)/gradientSquare,1e10) # less than line search math.pow(2,-20)
        #             #t = min(abs(fx)/gradientSquare,1.0/MM.sqrt(gradientSquare))
        #             #t = min(t,1e10)
        #             d = -t*gradient # more stable
        #             #d = -abs(fx)/gradientSquare*gradient
        #     else:
        #         # flat area. graident0 and cure0
        #         #d = -1.0*gradient/np.sqrt(gradientSquare)
        #         H_d = HessMatrix +MM.identity(HessMatrix.shape[0])*self.hessianDamp
        #         d = abs(ML.getInverse(H_d))@gradient*(-1.0)
        # else:
        #     d = abs(ML.getInverse(HessMatrix))@gradient*(-1.0)
        H_d = HessMatrix
        if isDetH0:
            H_d = HessMatrix +MM.identity(HessMatrix.shape[0])*hessianDamp
            #find proper H_d
            det_H_d = ML.det(H_d)
            if math.isclose(det_H_d,0.0,abs_tol=epsilon*epsilon):
                Log.Debug(tag_msg,f"calMinD>>,isDetH0={isDetH0},det_H_d = {det_H_d}")
                H_d = HessMatrix +MM.identity(HessMatrix.shape[0])*1e-16
       
        d = abs(ML.getInverse(H_d))@gradient*(-1.0)
        Log.Debug(tag_msg,f"calMinD,isDetH0={isDetH0},isGradient0={isGradient0},isf0={isf0},detH={detH},hessianM={HessMatrix},d={d}")
        # return d 
        
            # 
        # if math.isclose(detH,0.0,abs_tol=self.epslion):
        # #     #d = -1.0*gradient
        #     if math.isclose(fx,0.0,abs_tol=self.skipGrad0Eps): # avoid when f(x)=0, but not the minimum
        #         d =-1.0*gradient #-1.0/gradientSquare*gradient
        #         Log.Debug(NewtonMsgTag,f"fx_close to 0,fx={detH},d=\n{d},gradient={gradient}\n")
        #     else:
        #         d = -abs(fx)/gradientSquare*gradient # more stable
        #     Log.Debug(NewtonMsgTag,f"detH close to 0,detH={detH},d=\n{d},HessMatrix={HessMatrix}")
        # else:
        #     H_d = HessMatrix + MM.identity(HessMatrix.shape[0])*self.hessianDamp*2
        #     #min(1, 0.5/abs(fx))
        #     d = abs(ML.getInverse(H_d))@gradient*(-1.0)
        return d

    def calMin(initX,funcTuple,newtonArgsTuple,*FuncGradArgs,customLineSearcher=None,histDataCollector=None):
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
        (calF,calFAndGradient,calHessianMatrix) = funcTuple
        (maxIteration,epsilon,skipGrad0Eps,hessianDamp) = newtonArgsTuple
        lineSh = customLineSearcher
   
        if lineSh is None:
            lineSh = ArmijoWolfeLineSearcher(calF,calFAndGradient)
        iterNum = 0
        X = initX
        fx = None
        N = len(X)
        lastd = None
        gradient = None
        hessM = None
        while iterNum < maxIteration:
            (fx,gradient) = calFAndGradient(X,*FuncGradArgs)
            if histDataCollector is not None:
                histDataCollector.append((X,fx,gradient))
            # if math.isclose(fx,0.0,abs_tol=self.epslion):
            #     Log.Info(NewtonMsgTag,f"found result root={X},iterNum={iterNum}\n")
            #     return (X,fx,gradient)
            
            gradientSquare= gradient.T@gradient
            hessM = calHessianMatrix(X,*FuncGradArgs)
            detH = ML.det(hessM)
             # avoid special saddle point 
            # if(detH==0):
            #     X = X + self.skipGrad0Eps# self.epslion
            #     iterNum+=1
            #     Log.Debug(NewtonMsgTag,f"skip for detH=0-> iterNum={iterNum},X={X},fx={fx},grad=\n{gradient}\n,HesssianMatrix=\n{hessM}")
            #     continue
           
            if math.isclose(gradientSquare,0.0,abs_tol=epsilon*epsilon):
                # avoid saddle point and some other flat areas
                # detH <0 and grad ==0,   # normal saddle point z=x^2-y^2
                # detH~0, and grad = 0, might flat area or degenerated saddle point z=x^4-y^4
                Log.Debug(tag_msg,f"try find the min-> iterNum={iterNum},detH={detH},X={X},fx={fx},grad=\n{gradient}\n,HesssianMatrix=\n{hessM}\n,lastd={lastd}\n")
                
                if(fx == -1.0*math.inf): # already to the least
                    return (X,fx,gradient)
                
                # if(detH<=self.skipGrad0Eps):#self.epslion):# skipGrad0Eps
                #     Log.Debug(NewtonMsgTag,f"saddle or other situation")
                #     if lastd is None:
                #          X = X + self.skipGrad0Eps
                #     else:
                #         dt=lastd*1.0/(math.sqrt(lastd.T@lastd))
                #         X = X + self.skipGrad0Eps*dt# self.epslion
                #     iterNum+=1
                #     continue
                # Log.Debug(NewtonMsgTag,f"try find the min-> succeed-min")
                # return (X,fx,gradient)
            #d = -fx/gradient
            d = NewtonMinST._calMinD(fx,gradient,hessM,gradientSquare,detH,epsilon,skipGrad0Eps,hessianDamp)
            Log.Debug(tag_msg,f"iterNum={iterNum},X={X},fx={fx},detH={detH},d={d},grad=\n{gradient}\n,HesssianMatrix=\n{hessM}")
            alpha = lineSh.lineSearchMin(X,d,fx,gradient,*FuncGradArgs)
            # min(1, 0.5/abs(fx))
            X = X +d*alpha
            Log.Debug(tag_msg,f"iterNum={iterNum}-->,newX={X} oldX={X-d*alpha} d={d},alpha={alpha}")
            iterNum+=1
            lastd = d
           
        Log.Warning(tag_msg,f"Reached Maximum iterations X={X},NotFoundRoots,f={fx},f'={gradient}\n")
        return (X,fx,gradient)