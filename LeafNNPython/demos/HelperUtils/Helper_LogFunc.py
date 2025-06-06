tag_msg = "Helper_LogFunc"
class Helper_LogFunc:
    def logFuncLinear(X,argsList):
        """
        X will be vector [[x0],[x1],[x2]....] N*1
        [[u,c],[w,b]]
        u = argsList[0][0] :scalar
        c = argsList[0][1] :scalar
        w = argsList[1][0] :will be vector [[w0],[w1],[w2],..] N*1
        b = argsList[1][1] :scalar
    
        f = u*log(W*X+b)+c :scalar
        """
        W = argsList[0][0]
        b = argsList[0][1]
        u = argsList[1][0]
        c = argsList[1][1]
        base = argsList[1][2]
        Xl = W.T@X +b
        # check domain 
        f = None
        if Xl==0.0: # inf
            Log.Info(tag_msg,f"logFuncLinear->Xl={Xl}=0,the f will be -inf")
            f =  -1.0*u*math.inf
        elif Xl<0.0:
            Log.Error(tag_msg,f"logFuncLinear->invalid domain, Xl={Xl},which should >0,W={W},X={X},b={b}")
        else:
            f = u*math.log(Xl,base)+c
        return f

    def logFuncLinearGrad(X,argsList):
        """
        
        X will be vector [[x0],[x1],[x2]....] N*1
        [[u,c,base],[w,b]]
        u = argsList[0][0] :scalar
        c = argsList[0][1] :scalar
        base =argsList[0][2] :scalar
        w = argsList[1][0] :will be vector [[w0],[w1],[w2],..] N*1
        b = argsList[1][1] :scalar
    
        f = u*log(W*X+b)+c :scalar
        f'= u/[(W*X+b)*ln(base)]*W 
        """
        W = argsList[0][0]
        b = argsList[0][1]
        u = argsList[1][0]
        c = argsList[1][1]
        base = argsList[1][2]
        Xl = W.T@X +b
        # check domain 
        f,grad = None,None
        if Xl==0.0: # inf
            Log.Info(tag_msg,f"logFuncLinearGrad->Xl={Xl}=0,the f will be -inf")
            f =  -1.0*u*math.inf
            grad = MM.zeros(W.shape)
        elif Xl<0.0:
            Log.Error(tag_msg,f"logFuncLinearGrad->invalid domain, Xl={Xl},which should >0,W={W},X={X},b={b}")
        else:
            grad = u/(Xl*math.log(base))*W
            f = u*math.log(Xl,base)+c
        return (f,grad)

    def logLinearHessian(X,argsList):
        """
        X will be vector [[x0],[x1],[x2]....] N*1
        [[u,c,base],[w,b]]
        u = argsList[0][0] :scalar
        c = argsList[0][1] :scalar
        base =argsList[0][2] :scalar
        w = argsList[1][0] :will be vector [[w0],[w1],[w2],..] N*1
        b = argsList[1][1] :scalar
    
        f = u*log(W*X+b)+c :scalar
        f'= u/[(W*X+b)*ln(base)]*W 
        df''/(dxidxj) = u*wi*(WX+b)^-2*wj/ln(base)
        """
        W = argsList[0][0]
        b = argsList[0][1]
        u = argsList[1][0]
        c = argsList[1][1]
        base = argsList[1][2]
        Xl = W.T@X +b
        N = len(W)
        HessianM = MM.zeros([N,N])
        for i in range(N):
            for j in range(N):
                HessianM[i,j] = -1.0*u*W[i,0]*W[j,0]/(Xl*Xl*math.log(base))
        return HessianM

