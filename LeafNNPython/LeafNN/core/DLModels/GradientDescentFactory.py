import copy
import numpy as np 
import LeafNN.core.DLModels.TrainOptions as tp
from LeafNN.utils.Log import Log
from LeafNN.utils.Log import LogTag
class GradientDescentFactory:
    """
    2 dimensional array: scalar times vector
    """
    def layersMulti(grads,valMulti):
        """
        grads[0] : dJdw
        grads[1] : dJdb
        grads[0][0] dJdw[0]: the derivative of first layer
        """
        new_grads = []
        for i in range(2):
            dJ = []
            layers =len(grads[i])
            for j in range(layers):
                dJ.append(valMulti*grads[i][j])
            new_grads.append(dJ)
        return new_grads
    
    """
     2 dimensional array : vector dot
    """
    def layersDot(p,grad):
        result = 0.0
        for i in range(len(grad)):
            # i = 0 : grad[0]:DjDweights 
            # i = 1 : grad[1]:dJ/Dbias
            layers = len(grad[i])
            for l in range(layers):
                result =result + np.dot(np.transpose(p[i][l]),grad[i][l])
        return result
    """
    2 dimensional array : vector Add 
    """
    def layersPlus(grads,P):
        result = []
        for i in range(len(grads)):
            dJ = []
            for j in range(len(grads[i])):
                dJ.append(P[i][j]+grads[i][j])
            result.append(dJ)
        return result

    """
    modelLayersNum: how many layers 
    dJdw : derivative of cost to weights, current dJdw 
    dJdb : derivative of cost to bias, current dJdb
    output , weights: model weights 
    output , bias: model bias     
    """
    def SimpleBatchGradient(modelLayersNum,initLearnRate,dJdw,dJdb,weights,bias):
        print("simpleBatchGradient")
        for l in range(modelLayersNum-1):
            print(f"debug:l={l},dldw_shape={dJdw[l].shape},weightsShape={weights[l].shape}")
            print(f"debug: l={l},derivB shape={dJdb[l].shape}")
            #self.__normalizedGradients(self.derivLW,l)
            weights[l] += -1.0*initLearnRate*dJdw[l]
            bias[l] += -1.0*initLearnRate*dJdb[l] 
            print(f"l={l} derivLW>>>")
            print(dJdw[l])
            print(f"l={l} derivLB>>>")
            print(dJdb[l])
            print(f"l={l},weight>>")
            print(weights[l])
            print(f"l={l},bias>>")
            print(bias[l])
    


    """
    P : search direction p = -dJ1/dwn
    J2 : new Cost J(wn + alpha*P)
    J1 : current Cost J(wn)
    grad1: current gradient dJ1/dwn 
    alpha: learning rate 1.0>alpha>0
    armijo: f(wn+apha*P)<f(wn) + alpha*c1*p*grad1   alpha*c1*p*grad1 < 0, 
    """
    def FitArmijoCondition(P,c1,J2,J1,grad1,alpha):
        print(f"FitArmijoCondition>>> grad1 dldw={grad1[0][0]}, dldb={grad1[1][0]} J1={J1},alpha={alpha} J2={J2}")
        if(J2<J1+alpha*c1*GradientDescentFactory.layersDot(P,grad1)):
            return True
        else:
            return False
        
    """
    P : search direction p = -dJ1/dwn
    J2 : new Cost J(wn + alpha*P)
    J1 : current Cost J(wn)
    grad1: current gradient dJ1/dwn 
    alpha: learning rate 1.0>alpha>0
    WolfeCondition1: f(wn+apha*P)<f(wn) + alpha*c1*p*grad1   alpha*c1*p*grad1 < 0, 
    """
    def FitWolfeCondition1(P,c1,J2,J1,grad1,alpha):
        return GradientDescentFactory.FitArmijoCondition(P,c1,J2,J1,grad1,alpha)
    
    """
    P : search direction p = -dJ1/dwn
    J2 : new Cost J(wn + alpha*P)
    J1 : current Cost J(wn)
    grad1: current gradient dJ1/dwn 
    grad2: new gradient dJ2/dwn 
    WolfeCondition2: 
    c1<c2<1.0
    phi'(alpha)>c2*phi'(alpha=0)
    phi'(alpha) = dJ(wn + alpha*P)/dAlpha =  dJ(wn + alpha*P)/dwn*P
    phi'(alpha=0) = dJ(wn)/dwn*P
    grad2*P > c2*grad1*P
    """
    def FitWolfeCondition2(P,c2,grad1,grad2):
        if(GradientDescentFactory.layersDot(grad2,P)>c2*GradientDescentFactory.layersDot(grad1,P)):
            return True
        else:
            return False

    """
    P : search direction p = -dJ1/dwn
    J2 : new Cost J(wn + alpha*P)
    J1 : current Cost J(wn)
    grad1: current gradient dJ1/dwn 
    alpha: learning rate 1.0>alpha>0
    c1: 0.0<c1<1.0
    c2: c1<c2<1.0
    WolfeCondition1: J2<J1 + alpha*c1*p*grad1   alpha*c1*p*grad1 < 0, 
    WolfeCondition2:  grad2*P > c2*grad1*P   
    """ 
    def FitWolfConditions(P,c1,c2,J1,J2,grad1,grad2,alpha):
        if(GradientDescentFactory.FitWolfeCondition1(P,c1,J2,J1,grad1,alpha) and GradientDescentFactory.FitWolfeCondition2(P,c2,grad1,grad2)):
            return True
        else:
            return False
        
        
    def LineSearchWithWolfe(wb,getCostFunc,getGradFunc,J1,grad1,P,maxSearch,c1,c2,initAlpha,iteralNum):
        print("wolfe")
        alpha = initAlpha
        gradPower2 = GradientDescentFactory.layersDot(grad1,grad1)
        alpha = 1/(1+gradPower2)
        gradLength = np.sqrt(gradPower2)
      
        J2 = 0
        grad2 = []
        wb_new = []
        step = 0
        while (step < maxSearch):
            step +=1
            deltWb =  GradientDescentFactory.layersMulti(P,alpha)
            wb_new = GradientDescentFactory.layersPlus(wb,deltWb)
            [J2,cacheA,cacheZ] = getCostFunc(wb_new)
            grad2 = getGradFunc(cacheA,cacheZ)
            print(f"LineSearchWithWolfe_ trainWithLineSearch>> J1={J1},grads1 dldw={grad1[0][0]},grads1 dldb={grad1[1][0]} gradPower2={gradPower2}  p={P[0][0]},p_2={P[1][0]}")
            print(f"LineSearchWithWolfe_ trainWithLineSearch>> J2={J2},grads2 dldw={grad2[0][0]},grads2 dldb={grad2[1][0]}")
            if(not GradientDescentFactory.FitWolfeCondition1(P,c1,J2,J1,grad1,alpha)):
                alpha = alpha*0.5
                continue
            if(GradientDescentFactory.FitWolfeCondition2(P,c2,grad1,grad2)):
                print(f"succeed alpha=",alpha)
                break
            alpha = 0.9*alpha
            
           
        return [wb_new,grad2,J2,alpha]

  

    def BatchGradientWithLineSearch(wb,getCostFunc,getGradFunc,J1,grad1,iteralNum,lastGrad):
        """
        return [wb,grads,J2,alpha] 
        wb[0]:weights,wb[1]:bias, 
        grads[0]:dJdw, grads[1]:dJdb
        J2: new cost J(wn+alpha*P)
        alpha: proper learning rate
        """
        initLearnRate = 0.5
        print("BatchGradientWithLineSearch")
        print(f"BatchGradientWithLineSearch_ trainWithLineSearch>> J1={J1},grads dldw={grad1[0][0]},grads dldb={grad1[1][0]}")
        maxSearch = 15
        step = 1
        c1 = 0.1
        c2 = 0.9
        initAlpha = 0.8
          # normalize P
        P0 = GDF.layersMulti(grad1,-1.0)
        delt = GDF.layersPlus(grad1,GDF.layersMulti(lastGrad,-1.0))
        beta = GDF.layersDot(lastGrad,delt)/GDF.layersDot(lastGrad,lastGrad)
        P = GDF.layersPlus(P0,GDF.layersMulti(lastGrad,beta))
        return GDF.LineSearchWithWolfe(wb,getCostFunc,getGradFunc,J1,grad1,P,maxSearch,c1,c2,initAlpha,iteralNum)
    
    # only for 1 layer testing
    def packWB(weights,bias):
        return np.vstack([bias[0],weights[0]])
    
    def unpackWB(wb):
        b = wb[0:1,:]
        w = wb[1:,:]
        return [[w],[b]]


    def dot(a,b):
        return np.dot(np.transpose(a),b)
    
    
    def fmincg(f,X,options:tp.TrainOption,*args):
        if options :
            length = options.MaxIteration
        else:
            length = 100
        RHO = 0.01
        SIG = 0.5
        INT = 0.1
        EXT = 3.0
        MAX = 20
        RATIO = 100
        #argstr = f"f(X, {', '.join(str(arg) for arg in args)})"
        if isinstance(length, tuple):
            red = length[1]
            length = length[0]
        else:
            red = 1
        #S = 'Iteration '
        i = 0
        ls_failed = 0
        fX = []
        rates = []
        grads = []
        [f1, df1] = f(X, *args)
        fX.append(f1)
        i += length < 0
        s = -df1
        d1 = -1.0*GradientDescentFactory.dot(s,s)
        z1 = red/(1-d1)
        while i < abs(length):
            i += length>0
            X0 ,f0,df0 = X,f1,df1
            X = X + z1*s
            rates.append(z1)
            grads.append(df1)
            if(i == 7):
                Log.Debug("Key_Step",f"key steps::{X}")
                #X =np.array([[98.05141701],[-0.78685205], [-0.96436821]])
            [f2,df2] = f(X,*args)
            i += length<0
            d2 = GradientDescentFactory.dot(df2,s)
            f3,d3,z3 = f1,d1,-z1
            if length>0:
                M = MAX
            else:
                M = min(MAX,-length-i)
            success = 0
            limit = -1
            Log.Debug("testTrain","fmincg>>")
            Log.Debug("testTrain",f"beginSearch i={i} f1/f0={f1},f2/f1={f2} z1/a0={z1} d1/d0={d1} d2/d1={d2} s0={s} df2/df1={df2} X={X}")

            # line searching
            while True:
                # search1 learning rate z when breaking wolfe condition
                while(((f2 > f1 + z1*RHO*d1) or (d2 > -SIG*d1)) and M>0):
                    limit = z1
                    if f2 > f1:
                        z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3)
                        Log.Debug("testTrain",f"here0 i={i} j={MAX-M}  z2/a1={z2},z3/a2={z3}")
                    else:
                        A = 6*(f2-f3)/z3 + 3*(d2+d3)
                        B = 3*(f3-f2)-z3*(d3+2*d2)
                        z2 = (np.sqrt(B*B-A*d2*z3*z3)-B)/A
                        Log.Debug("testTrain",f"here1 i={i} j={MAX-M} A={A},B={B} z2/a1={z2},z3/a2={z3},d3/d2={d3}")
                    if np.isnan(z2) or np.isinf(z2):
                        z2 = z3/2.0
                    z2 = max(min(z2,INT*z3),(1-INT)*z3)
                    z1 = z1 + z2
                    X = X + z2*s 
                    [f2,df2] = f(X,*args)
                    Log.Debug("testTrain",f"here2 i={i} j={MAX-M} f2/f1={f2},f1/f0={f1},df2/df1={df2},z1/a0={z1},z2/a1={z2}")
                    M = M - 1
                    i += length<0
                    d2 = GradientDescentFactory.dot(df2,s)
                    z3 = z3 - z2
                    Log.Debug("testTrain",f"here3 i={i} j={MAX-M-1} z3/a2={z3}")
                # search1 end
                if f2 > f1+z1*RHO*d1 or d2> -SIG*d1:
                    break # failed 
                elif d2> SIG*d1:
                    success = 1
                    break #succeed
                elif M == 0:
                    break #failed out of time
                A = 6*(f2-f3)/z3+3*(d2+d3)
                B = 3*(f3-f2)-z3*(d3+2*d2)
                Log.Debug("testTrain",f"right_here4 i={i} j={MAX-M},A={A},B={B} f2/f1={f2},f3/f2={f3},z1/a0 ={z1} d1/d0 = {d1} c1={RHO} c2={SIG} z3/a2={z3},d2/d1={d2},d3/d2={d3}")
                z2 = -d2*z3*z3/(B+np.sqrt(B*B-A*d2*z3*z3))
                Log.Debug("testTrain",f"right_here5 i={i} j={MAX-M},update z2/a1 ={z2}")
                if not np.isreal(z2) or np.isnan(z2) or np.isinf(z2) or z2<0:
                    if limit < -0.5:
                        z2 = z1*(EXT-1.0)
                    else:
                        z2 = (limit-z1)/2.0
                elif (limit>-0.5) and ((z2+z1)>z1*EXT):
                    z2 = (limit-z1)/2.0
                elif (limit<-0.5) and  ((z2+z1)>z1*EXT):
                    z2 = z1*(EXT-1.0)
                elif (limit<-0.5) and (z2<((limit-z1)*(1.0-INT))):
                    z2 = (limit-z1)*(1.0-INT)
                f3,d3,z3 = f2,d2,-z2
                z1 = z1 +z2
                X = X + z2*s 
                [f2,df2] = f(X,*args)
                M = M-1
                i += length<0
                d2 = GradientDescentFactory.dot(df2,s)
            # go on updating searching direction based on line-searching result
            if success:
                f1 = f2
                fX.append(f1)
                s = (GradientDescentFactory.dot(df2,df2)-GradientDescentFactory.dot(df1,df2))/GradientDescentFactory.dot(df1,df1)*s - df2
                tmp = df1
                df1 = df2
                df2 = tmp
                d2 = GradientDescentFactory.dot(df1,s)
                Log.Debug("testTrain",f"here6 i={i} j={MAX-M},newf={f1}, success_s/s1 ={s},d2/d1={d2},df1/df0={df1},df2/df1={df2},z1/a0={z1}")
                if d2 > 0:
                    s = -df1
                    d2 = -1.0*GradientDescentFactory.dot(s,s)
                z1 = z1 * min(RATIO,d1/(d2 - np.finfo(float).eps))
                d1 = d2
                ls_failed = 0
                Log.Debug("testTrain",f"here7 after:i={i} j={MAX-M} z1/a0={z1},d1/d0={d1},z2/a1={z2},z3/a2={z3}")
            else:
                X , f1, df1 = X0,f0,df0
                Log.Debug("testTrain","failed") 
                if ls_failed or i > abs(length):
                    if ls_failed:
                        Log.Warning(LogTag.DLModels,f"fmincg>>failed, Failed twice")
                    break       # failed for fmincg  (line Search failed twice or out of maxIteration)
                tmp = df1
                df1 = df2 
                df2 = tmp
                s = -df1
                d1 = -1.0*GradientDescentFactory.dot(s,s)
                z1 = 1.0/(1.0-d1)
                ls_failed = 1
        return[X,fX,i,rates,grads]

    """
    1. Use line-search to get proper learning rate based on wolfe conditions
    2. use Polar to get search direction
    Inputs:
    X : variables of f, to get the minimum.  (usually model weights,bias)
    f : return cost and gradient at current X, convex is necessary
    *args: training data which required by f
    options: training options 
    """
    def fminWithPolar(f,X_input,options:tp.TrainOption,*args):
        mEnable = options.monitorOption.enable
        maxIter = options.MaxIteration
        mtdata = None
        if(mEnable):
            mtdata = tp.MonitorData()
        c1 = options.C1
        c2 = options.C2
        EXT =options.EXT
        INT = options.INT
        RATIO = options.RATIO
        X = X_input
        
        [f0,df0] = f(X,*args)
        s0 = -df0
        d0 = -1.0*GDF.dot(s0,s0)
        a0 = 1.0/(1.0-d0)
        ls_failed = 0 
        
        if(options.monitorOption.enable):
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
            if(i == 7):
                #Log.Debug("Key_Step",f"key steps::shape={X.shape} value: {repr(X[0][0])},{repr(X[1][0])},{repr(X[2][0])}")
                #X =np.array([[98.05141701],[-0.78685205], [-0.96436821]])
                Log.Debug("Key_Step",f"after shape={X.shape} X= {X}")
            [f1,df1] = f(X,*args)
            d1 = GDF.dot(df1,s0)
            # 2. line-search to get a (learning rate) 
            # wolfe1: f2<f1+c1*a*df1*s1    
            # wolfe2: df2*s1>c2*df1*s1   df2*s1<-c2*df1*s1
            j = 0
            succeed = False
            f2,d2,a2 = f0,d0,-a0
            limit = -1
            
            Log.Debug("testTrain","fminWithPolar>>")
            Log.Debug("testTrain",f"beginSearch i={i} f1/f0={f0},f2/f1={f1} z1/a0={a0} d1/d0={d0} d2/d1={d1} s0={s0} df2/df1={df1} X={X}")

            while j <options.MaxLineSearch:
                nanCost = np.isnan(f1) or np.isnan(f0)
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
                        a1 = (np.sqrt(B*B-A*d1*a2*a2)-B)/A
                        Log.Debug("testTrain",f"here1 i={i} j={j} A={A},B={B} z2/a1={a1},z3/a2={a2},d3/d2={d2}")
                    if np.isnan(a1) or np.isinf(a1):
                        a1 = a2/2.0
                    a1 = max(min(a1,INT*a2),(1-INT)*a2)
                    a0 = a0 + a1
                    X = X + a1*s0
                    [f1,df1] = f(X,*args)
                    Log.Debug("testTrain",f"here2 i={i} j={j} f2/f1={f1},f1/f0={f0},df2/df1={df1},z1/a0={a0},z2/a1={a1}")
                    d1 = GDF.dot(df1,s0)
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
                    a1 = -d1*a2*a2/(B+np.sqrt(B*B-A*d1*a2*a2))
                    Log.Debug("testTrain",f"right_here5 i={i} j={j},update z2/a1 ={a1}")
                    if not np.isreal(a1) or np.isnan(a1) or np.isinf(a1) or a1<0:
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
                    d1 = GDF.dot(df1,s0)
                    # if(nanCost):
                    #     succeed = True
                j = j+1
            if (succeed):
                f0 = f1
                # 1. update search directions with Polar
                # s1 = -1.0*df1 + beta*s0
                #beta = df1*(df1-df0)/(df0*df0) why? it's not right 
                #beta = GDF.dot(df1,(df1-df0))/GDF.dot(df0,df0)
                beta =( GDF.dot(df1,df1) - GDF.dot(df1,df0) ) /GDF.dot(df0,df0)
                s1 =-1.0*df1 + beta*s0
                #GDF.swap(df0,df1)
                #df0,df1 = df1,df0
                tmp = df0
                df0 = df1
                df1 = tmp
                d1 =  GDF.dot(df0,s1)
                Log.Debug("testTrain",f"here6 i={i} j={j},newf={f0}, success_s/s1 ={s1},d2/d1={d1},df1/df0={df0},df2/df1={df1},z1/a0={a0}")
                if(d1>0):
                    s1 = -1.0*df0
                    d1 = -1.0*GDF.dot(s1,s1)
                a0 = a0 *min(RATIO,d0/(d1-np.finfo(float).eps))  # it will faster its converge
                d0 = d1
                Log.Debug("testTrain",f"here7 after:i={i} j={j} z1/a0={a0},d1/d0={d0},z2/a1={a1},z3/a2={a2}")
                ls_failed = 0   
            else:
                X,f0,df0 = Xb,fb,dfb
                Log.Debug("testTrain","failed") 
                if ls_failed:
                    Log.Warning(LogTag.DLModels,"fminWithPolar>>Failed twice")    
                    break
                #GDF.swap(df0,df1)
                #df0,df1 =df1,df0
                tmp = df0
                df0 = df1
                df1 = tmp
                s1 = -df0
                d0 = -1.0*GDF.dot(s1,s1)
                a0 = 1.0/(1.0 - d0)
                ls_failed = True
            
            if(options.monitorOption.enable):
                mtdata.costs.append(f0)
                mtdata.grads.append(df0)
                mtdata.iterationInds.append(i)
                mtdata.rates.append(a0)
                mtdata.sucesses.append(succeed)

            s0 = s1 # todp
            i = i+1
        return[X,mtdata]

        

GDF = GradientDescentFactory