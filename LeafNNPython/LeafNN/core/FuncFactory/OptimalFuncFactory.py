from LeafNN.Bases.MathMatrix import MathMatrix as MM
#import numpy as np
from LeafNN.core.LeafModels.TrainOptions import TrainOption as TOps
import LeafNN.core.LeafModels.TrainMonitor as TMot
from LeafNN.utils.Log import Log
OptimalTag = "OptimalTag"
class OptimalFuncFactory:
    """
    1. Use line-search to get proper learning rate based on wolfe conditions
    2. use Polar to get search direction
    Inputs:
    X : variables of f, to get the minimum.  (usually model weights,bias)
    f : return cost and gradient at current X, convex is necessary
    *args: training data which required by f
    options: training options 
    """
    def OptimalMinWithWolfes(f,X_input,trainOption:TOps,monitorOption:TMot.MonitorOption=None,*args):
        
        mEnable =False
        if(monitorOption and monitorOption.enable):
            mEnable = monitorOption.enable
        maxIter = trainOption.MaxIteration
        mtdata = None
        if(mEnable):
            mtdata = TMot.MonitorData()
        c1 = trainOption.C1
        c2 = trainOption.C2
        EXT =trainOption.EXT
        INT = trainOption.INT
        RATIO = trainOption.RATIO
        X = X_input
        
        [f0,df0] = f(X,*args)
        s0 = -df0
        #d0 = -1.0*GDF.dot(s0,s0)
        d0 = -1.0*(s0*s0)
        a0 = 1.0/(1.0-d0)
        ls_failed = 0 
        
        if(mEnable):
            mtdata.costs.append(f0)
            mtdata.grads.append(df0)
            mtdata.iterationInds.append(0)
            mtdata.rates.append(a0)
            mtdata.sucesses.append(True)
        i = 1
        a1 = 0.0#initial 
        s1 = s0 # global
        while i <= maxIter:
            # back current states
            Xb,fb,dfb = X,f0,df0  

            X = X + a0*s0
            # if(i == 7):
            #     #Log.Debug("Key_Step",f"key steps::shape={X.shape} value: {repr(X[0][0])},{repr(X[1][0])},{repr(X[2][0])}")
            #     #X =np.array([[98.05141701],[-0.78685205], [-0.96436821]])
            #     Log.Debug("Key_Step",f"after shape={X.shape} X= {X}")
            [f1,df1] = f(X,*args)
            #d1 = GDF.dot(df1,s0)
            d1 = df1*s0
            # 2. line-search to get a (learning rate) 
            # wolfe1: f2<f1+c1*a*df1*s1    
            # wolfe2: df2*s1>c2*df1*s1   df2*s1<-c2*df1*s1
            j = 0
            succeed = False
            f2,d2,a2 = f0,d0,-a0
            limit = -1
            
            Log.Debug("testTrain","fminWithPolar>>")
            Log.Debug("testTrain",f"beginSearch i={i} f1/f0={f0},f2/f1={f1} z1/a0={a0} d1/d0={d0} d2/d1={d1} s0={s0} df2/df1={df1} X={X}")

            while j <trainOption.MaxLineSearch:
                nanCost = MM.isnan(f1) or MM.isnan(f0)
                # = is for f=inf 
                wolfe1 = f1<=f0+c1*a0*d0 
                wolfe2_lower = d1>=c2*d0
                wolfe2_upper = d1<=-c2*d0
                if( (not wolfe1 and not nanCost) or not wolfe2_upper) :
                    limit = a0
                    if(f1>f0):
                        a1 = a2-(0.5*d2*a2*a2)/(d2*a2+f1-f2)
                        Log.Debug("testTrain",f"here0 i={i} j={j}  z2/a1={a1},z3/a2={a2}")
                    else:
                        A = 6*(f1-f2)/a2+3*(d1+d2)
                        B = 3*(f2-f1)-a2*(d2+2*d1)
                        a1 = (MM.sqrt(B*B-A*d1*a2*a2)-B)/A
                        Log.Debug("testTrain",f"here1 i={i} j={j} A={A},B={B} z2/a1={a1},z3/a2={a2},d3/d2={d2}")
                    if MM.isnan(a1) or MM.isinf(a1):
                        a1 = a2/2.0
                    a1 = max(min(a1,INT*a2),(1-INT)*a2)
                    a0 = a0 + a1
                    X = X + a1*s0
                    [f1,df1] = f(X,*args)
                    Log.Debug("testTrain",f"here2 i={i} j={j} f2/f1={f1},f1/f0={f0},df2/df1={df1},z1/a0={a0},z2/a1={a1}")
                    #d1 = GDF.dot(df1,s0)
                    d1 = df1*s0
                    a2 = a2 -a1
                    Log.Debug("testTrain",f"here3 i={i} j={j} z3/a2={a2}")
                    #
                # inf case, wolf1 or nanCost  
                elif(wolfe2_lower and wolfe2_upper):
                    succeed = True
                    break
                else:        #(not wolfe2_lower) or nanCost:
                    A = 6.0*(f1-f2)/a2+3.0*(d1+d2)
                    B = 3.0*(f2-f1)-a2*(d2+2*d1)
                    Log.Debug("testTrain",f"right_here4 i={i} j={j},A={A},B={B} f2/f1={f1},f3/f2={f2},z1/a0 ={a0} d1/d0 = {d0} c1={c1} c2={c2} z3/a2={a2},d2/d1={d1},d3/d2={d2}")
                    a1 = -d1*a2*a2/(B+MM.sqrt(B*B-A*d1*a2*a2))
                    Log.Debug("testTrain",f"right_here5 i={i} j={j},update z2/a1 ={a1}")
                    if not MM.isreal(a1) or MM.isnan(a1) or MM.isinf(a1) or a1<0:
                        if limit < -0.5:
                            a1 = a0*(EXT-1.0)
                        else:
                            a1 = (limit-a0)/2.0
                    elif (limit>-0.5) and ((a1+a0)>a0*EXT):
                        a1 = (limit-a0)/2.0
                    elif (limit<-0.5) and  ((a1+a0)>a0*EXT):
                        a1 = a0*(EXT-1.0)
                    elif (limit<-0.5) and (a1<((limit-a0)*(1.0-INT))):
                        a1 = (limit-a0)*(1.0-INT)
                    f2,d2,a2 = f1,d1,-a1
                    a0 = a0 + a1
                    X = X + a1*s0
                    [f1,df1] = f(X,*args)
                    #d1 = GDF.dot(df1,s0)
                    d1 = df1*s0
                    # if(nanCost):
                    #     succeed = True
                j = j+1
            if (succeed):
                f0 = f1
                # 1. update search directions with Polar
                # s1 = -1.0*df1 + beta*s0
                #beta = df1*(df1-df0)/(df0*df0) why? it's not right 
                #beta = GDF.dot(df1,(df1-df0))/GDF.dot(df0,df0)
                #beta =( GDF.dot(df1,df1) - GDF.dot(df1,df0) ) /GDF.dot(df0,df0)
                beta =((df1*df1) - (df1*df0))/(df0*df0)
                s1 =-1.0*df1 + beta*s0
                #GDF.swap(df0,df1)
                #df0,df1 = df1,df0
                tmp = df0
                df0 = df1
                df1 = tmp
                #d1 =  GDF.dot(df0,s1)
                d1 = df0*s1
                Log.Debug("testTrain",f"here6 i={i} j={j},newf={f0}, success_s/s1 ={s1},d2/d1={d1},df1/df0={df0},df2/df1={df1},z1/a0={a0}")
                if(d1>0):
                    s1 = -1.0*df0
                    #d1 = -1.0*GDF.dot(s1,s1)
                    d1 = -1.0*(s1*s1)
                a0 = a0 *min(RATIO,d0/(d1-MM.finfo(float).eps))  # it will faster its converge
                d0 = d1
                Log.Debug("testTrain",f"here7 after:i={i} j={j} z1/a0={a0},d1/d0={d0},z2/a1={a1},z3/a2={a2}")
                ls_failed = 0   
            else:
                X,f0,df0 = Xb,fb,dfb
                Log.Debug("testTrain","failed") 
                if ls_failed:
                    Log.Warning(OptimalTag,"fminWithPolar>>Failed twice") 
                    # todo   
                    break
                #GDF.swap(df0,df1)
                #df0,df1 =df1,df0
                tmp = df0
                df0 = df1
                df1 = tmp
                s1 = -df0
                #d0 = -1.0*GDF.dot(s1,s1)
                d0 = -1.0*(s1*s1)
                a0 = 1.0/(1.0 - d0)
                ls_failed = True
            
            if(mEnable):
                mtdata.costs.append(f0)
                mtdata.grads.append(df0)
                mtdata.iterationInds.append(i)
                mtdata.rates.append(a0)
                mtdata.sucesses.append(succeed)

            s0 = s1 # todp
            i = i+1
        return[X,mtdata]

        