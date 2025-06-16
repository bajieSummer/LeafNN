from LeafNN.utils.Log import Log
from LeafNN.ConvexOptimizer.LineSearcher import ArmijoLineSearcher
from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.Bases.MatrixLinear import MatrixLinear as ML
from .GradientDescent import GradientDescent
import math
tag_msg="GradientDescentMinAdam"
class GradientDescentAdam(GradientDescent):
    def __init__(self,calFFunc,calFAndGradient,maxIteration=200,epslion =1e-20,alpha=0.001,beta1=0.9,beta2=0.999):
        super().__init__(calFFunc,calFAndGradient,maxIteration,epslion,alpha)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsAdd = 1e-8
        Log.Debug(tag_msg,f"GradientDescentAdam init")

    
    def calMin(self,initX,*FuncGradArgs,customLineSearcher=None,histDataCollector=None):
        """
        calMin of f,
        intX->first search point
        funcGradArgs-> parameters of for calF,calFAndGrad,calHessianFunc
        customLineSearcher-> if is None: then defaultLineSearcher = ArmijoWolfeLineSearcher
        return (X,fx,gradient,f'')
        """
        funcTuple = (self.calFFunc,self.calFuncAndGradient)
        GDArgsTuple = (self.maxIteration,self.epslion,self.alpha,self.beta1,self.beta2,self.epsAdd)
        return GradientDescentAdam.calMinGlobal(initX,funcTuple,newtonArgsTuple,*FuncGradArgs,customLineSearcher=customLineSearcher,histDataCollector=histDataCollector)
   

    def _calMinD(iterNum,gradient,gradientSquare,lastm,lastv,beta1,beta2,eps):
        # m_k+1 = beta1*m_k + (1.0-beta1)*g
        # v_k+1 = beta2*v_k + (1.0-beta2)*g^2
        # mh_k = m_k/(1-beta1**k)
        # vh_k = v_k/(1-bet2**k)
        # d = mh_k/(sqrt(vh_k)+ eps)
        k = iterNum
        if lastm is None or lastv is None:
            m = gradient
            v = gradientSquare
            d = m/(math.sqrt(v)+eps)
            return (d,m,v)
        m = lastm*beta1 +(1.0-beta1)*gradient
        v = lastv*beta2 +(1.0-beta2)*gradientSquare
        mh = m/(1.0-beta1**k)
        vh = v/(1.0-beta2**k)
        d = mh/(math.sqrt(vh)+eps)
        return (d,m,v)


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
        (maxIteration,epsilon,alphaRate,beta1,beta2,eps) = GDArgsTuple
        lineSh = customLineSearcher

        if lineSh is None:
            lineSh = ArmijoLineSearcher(calF,calFAndGradient)
        iterNum = 0
        X = initX
        fx = None
        N = len(X)

        m = None
        v = None 
        while iterNum < maxIteration:
            (fx,gradient) = calFAndGradient(X,*FuncGradArgs)
            if histDataCollector is not None:
                histDataCollector.append((X,fx,gradient))
        
            gradientSquare= gradient.T@gradient
            if(fx == -1.0*math.inf): # already to the least
                return (X,fx,gradient)
            (d,m,v) = GradientDescentAdam._calMinD(iterNum,gradient,gradientSquare,m,v,beta1,beta2,eps)
            

            
            d2 = alphaRate*d
            alpha = lineSh.lineSearchMin(X,d2,fx,gradient,*FuncGradArgs)
            # min(1, 0.5/abs(fx))
            Log.Debug(tag_msg,f"iterNum={iterNum},X={X},fx={fx},d={d},grad=\n{gradient}\n,alphaRate=\n{alphaRate}")
            X = X +alpha*alphaRate*d
            Log.Debug(tag_msg,f"iterNum={iterNum}-->,newX={X} oldX={X-d*alpha} d={d},alpha={alpha}")
            iterNum+=1
            lastd = d
        
        Log.Warning(tag_msg,f"Reached Maximum iterations X={X},not found min,f={fx},f'={gradient}\n")
        return (X,fx,gradient)