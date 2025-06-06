from LeafNN.utils.Log import Log
from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.Bases.MatrixLinear import MatrixLinear as ML
from .LineSearcher import ArmijoWolfeLineSearcher
import math
import random
tag_msg = "NewtonMinBFGS"
#https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
#
# BFGS
# sk = xk+1 - xk
# yk = gradk+1 - gradk
# Bk+1*sk = yk
# Bk+1 = Bk+alpha*u*u.T - beta*v*v.T  u = yk, alpha = 1.0/(yk.T*sk), v = Bk*sk, beta = 1.0/(sk.T*Bk*sk)
# Hk+1 = Vk.T*Hk*Vk+alpha*sk*sk.T  Vk = (I-alpha*yk*sk.T)
# dk+1 = -Hk+1*Gradk+1

class NewtonMinBFGS:
    def _calMinD(currentX,currentGrad,gradientSquare,lastX,lastGrad,lastH,epsilon,tflat,tEscapeSaddleTry):
        Log.Debug(tag_msg,f"calMinD-->")
        
        # return (d,H), 
        # d: current search direction, 
        # H: inv of current Hessian Matrix, 
        N = len(currentX)
        I = MM.identity(N)
        t1 = math.sqrt(gradientSquare)
        if lastX is None or lastGrad is None:
            # first
            #N = len(currentX)# N variables
            d = -I@currentGrad/t1
            return (d,I,tflat)
    
        sk = currentX - lastX
        yk = currentGrad-lastGrad
        t = yk.T@sk 
        
        ykSquare = yk.T@yk
        epsFlat=1e-5
        epsFlatSqur = epsFlat*epsFlat
        tEscapeSaddleTryStart = tEscapeSaddleTry//3
        # flat area or saddle point
        if math.isclose(0.0,gradientSquare,abs_tol=epsFlatSqur) and math.isclose(0.0,ykSquare,abs_tol=epsFlatSqur):
            #H = lastH
            # only quiet close to zero
            # if math.isclose(0.0,t1,abs_tol=epsilon*epsilon): 
            #     d = -currentGrad
            # if t1 == 0.0: # can't calculate any more
            #     d = -currentGrad
            tflat +=1
            if tflat >tEscapeSaddleTryStart: # try escape
                #only quiet close to zero
                if math.isclose(0.0,t1,abs_tol=epsilon*epsilon): 
                    d = -currentGrad
                else:
                    d = -0.01*currentGrad/t1
                Log.Debug(tag_msg,f"grad_curve_near_zero: currentX={currentX},tflat={tflat},d={d},gradSqaure={gradientSquare},tEscapeSaddleTry={tEscapeSaddleTry}")
                return (d,I,tflat)
            

        if gradientSquare>epsFlat and tflat>0:
            Log.Debug(tag_msg,f"escape_saddle_area: currentX={currentX},tflat={tflat}")
            # means skip current saddle point,so reset tflat
            tflat = 0

     
        alpha = 1.0
        if math.isclose(0.0,t,abs_tol=epsilon):
            alpha = 1.0/(t+epsilon)
        else:
            alpha = 1.0/t
        
        MS = sk@sk.T
        
        Vk = (I - alpha*yk@sk.T)
        delt = 1e-10
       
        H = Vk.T@lastH@Vk + alpha*MS
        if t<0.0 or math.isclose(t,0.0,abs_tol=epsilon):
            Log.Debug(tag_msg,f"negative or less 0 t={t}")
            H = lastH + delt*I # lastH+delt*I
        #calH
        Log.Debug(tag_msg,f"calMinD-->currentX={currentX}\n,lastX={lastX}\n,grad={currentGrad}\n,lastGrad={lastGrad}\n lastH={lastH}\n tflat={tflat} t={t}\n")

        #H = Vk.T@lastH@Vk + alpha*MS
        # detH = 1.0/ML.det(H)
        d = -H@currentGrad # why abs
        return (d,H,tflat)


    def calMin(initX,funcTuple,newtonArgsTuple,*FuncGradArgs,customLineSearcher=None,histDataCollector=None):
        Log.Debug(tag_msg,f"calMin")
        """
        calMin of f,
        intX->first search point
        funcTuple =(calF,calFAndGradient)
        funcGradArgs-> parameters of for calF,calFAndGradient,calHessianMatrix
        (maxIteration,epsilon,skipGrad0,hessianDamp) ->newtonArgsTuple
        customLineSearcher-> if is None: then defaultLineSearcher = ArmijoWolfeLineSearcher
        return (X,fx,gradient,f'')
        """
        (calF,calFAndGradient) = funcTuple
        (maxIteration,epsilon,tflatMax) = newtonArgsTuple
        lineSh = customLineSearcher
   
        if lineSh is None:
            lineSh = ArmijoWolfeLineSearcher(calF,calFAndGradient)
            lineSh.sigma1 = 0.1
            lineSh.sigma2 = 0.8
        iterNum = 0

        X = None
        N = len(initX)
        if isinstance(initX, list):
            X = MM.array(initX).reshape([N,1])
        else: # todo
            X = initX 
        fx = None
        
        gradient = None
        H = None
        tflat = 0
        lastX = None
        lastGrad = None
        
        # use EMA to escape saddle point
        beta1 = 0.9
        lastd = None
        tEscapeSaddleTry = tflatMax//5
        while iterNum < maxIteration:
            (fx,gradient) = calFAndGradient(X,*FuncGradArgs)
            if histDataCollector is not None:
                histDataCollector.append((X,fx,gradient))
    
            gradientSquare= gradient.T@gradient
            if(fx == -1.0*math.inf): # already to the least
                    Log.Debug(tag_msg,f"find the min-> -inf iterNum={iterNum},fx={fx},X={X},grad=\n{gradient}\n,InvHesssianMatrix=\n{H}\n")
                    return (X,fx,gradient)
            if math.isclose(gradientSquare,0.0,abs_tol=epsilon*epsilon):
                if lastX is None: #if first  skip first zero point
                    X =X +0.01
                    continue
                if(tflat > tflatMax):
                    Log.Debug(tag_msg,f"find the min-> flatMax iterNum={iterNum},tflatMax={tflatMax} tflax={tflat} fx={fx},X={X},grad=\n{gradient}\n,InvHesssianMatrix=\n{H}\n")
                    return (X,fx,gradient)
                if(tflat>tEscapeSaddleTry+1):
                    dx = X-lastX
                    dxSquare = dx.T@dx
                    if math.isclose(dxSquare,0.0,abs_tol=epsilon*epsilon*epsilon):# more strict
                        Log.Debug(tag_msg,f"find the min-> localMin iterNum={iterNum},tflat={tflat} fx={fx},X={X},grad=\n{gradient}\n,InvHesssianMatrix=\n{H}\n")
                        return (X,fx,gradient)

            (d,H,tflat) = NewtonMinBFGS._calMinD(X,gradient,gradientSquare,lastX,lastGrad,H,epsilon,tflat,tEscapeSaddleTry) 
            # d = NewtonMinBFGS._calMinD(fx,gradient,hessM,gradientSquare,detH,epsilon,skipGrad0Eps,hessianDamp)
            Log.Debug(tag_msg,f"beforeLS-iterNum={iterNum}-->fx={fx},d=\n{d}\n,X=\n{X}\n grad={gradient}\n")
            # if lastd !=None:
            #     d =  beta1*lastd + (1.0-beta1)*d
            alpha = lineSh.lineSearchMin(X,d,fx,gradient,*FuncGradArgs)
            Log.Debug(tag_msg,f"afterLS-iterNum={iterNum}-->fx={fx},alpha={alpha},oldX=\n{X}\n,newX={X+d*alpha}\n d=\n{d}\n,grad=\n{gradient}\n,InvHesssianMatrix=\n{H}\n")
            # min(1, 0.5/abs(fx))
            lastX = X 
            lastGrad = gradient
            lastd = d
            X = X +d*alpha
           
            iterNum+=1
        Log.Warning(tag_msg,f"Reached Maximum iterations X={X},NotFoundRoots,f={fx},f'={gradient}\n")
        return (X,fx,gradient)