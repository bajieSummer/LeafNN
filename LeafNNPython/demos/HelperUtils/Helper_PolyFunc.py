from LeafNN.Bases.MathMatrix import MathMatrix as MM
class PolyFuncHelper:
    def calDPolyF(X,argsList):
        # xi_argus = argsList[i]
        # for xi-> fxi += argsList[i][0] +  argsList[i][1]*Xi  + argsList[i][2]*Xi*Xi + argsList[i][3]*Xi*Xi*Xi + ..
        # f = fx1+fx2+...fxN
        # check shape:
    
        if(X.shape !=None and len(X.shape)!=2 and X.shape[1]!=1):
            Log.Error(tag_msg,f"shape is not right, we should have (N,1), but shapeX={X.shape}")
            return None
        cacheX = []
        f = 0
        N = len(argsList) # which means N variables
        cacheX = MM.ones([N,1])
        for i in range(N):
            M = len(argsList[i])# means the ithe variable has M polys

            for j in range(M): 
                if i==0 and j>0:
                    cacheX = MM.hstack([cacheX,X*cacheX[:,j-1:j]])
                arg = argsList[i][j]
                if arg!=0:
                    f+=arg*cacheX[i][j]
        return f

    def calDPolyFAndGrad(X,argsList):
        # N = len(argsList): which means N variables
        # xi_argus = argsList[i]
        # for xi-> fxi += argsList[i][0] +  argsList[i][1]*Xi  + argsList[i][2]*Xi*Xi + argsList[i][3]*Xi*Xi*Xi + ..
        # f = fx1+fx2+...fxN
        # grad[i] =  argsList[i][1] + argsList[i][2]*2*xi*xi + .. argsList[i][j]*j*xi^j+...
        # for xi cacheX[i] = [1,xi,xi^2,xi^3, ...]
        gradArr =[]
        f = 0
        N = len(argsList) # which means N variables
        cacheX = MM.ones([N,1])
        for i in range(N):
            M = len(argsList[i])# means the ithe variable has M polys
            gradArr.append(0)
            for j in range(M): 
                if i==0 and j>0:
                    #print(f"X^j=\n{X*cacheX[:,j-1:j]}")
                    cacheX = MM.hstack([cacheX,X*cacheX[:,j-1:j]])
                arg = argsList[i][j]
                #print(f"cacheX=\n{cacheX}")
                if arg!=0:f+=arg*cacheX[i][j]
                if j>0:
                    if arg!=0:gradArr[i]+=arg*cacheX[i][j-1]*j
        grad = MM.array(gradArr).reshape([N,1])
        return (f,grad)
    