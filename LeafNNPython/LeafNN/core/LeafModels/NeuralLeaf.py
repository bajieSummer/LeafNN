from LeafNN.Bases.MathMatrix import MathMatrix as MM
#import numpy as np
from LeafNN.utils.Log import Log
from LeafNN.core.LeafModels.Leaf import Leaf
NeuralLeafTag = "NeuralLeafTag"
class NeuralLeaf(Leaf):
    def __init__(self,matrixArray):
        super().__init__(matrixArray)
    
    def XAddOnes(X):
        ones_x = MM.ones([X.shape[0],1])*1.0
        X = MM.hstack([ones_x,X])
        return X
    
    """
    X_input: dataX
    activeFunc: active function
    return[Z,A] Z:Leaf, A:Leaf
    """
    def forward(self,X_input,activeFunc):
        #WX
        X = NeuralLeaf.XAddOnes(X_input)
        z_matrixs = [None]#first layer don't process
        a_matrixs = [X]
        success = True
        lastA = X
        layerSize = self.getLayerSize()
        l = 0
        for mat in self._matrixs:
            #todo why would it save time of training? and make it converge, (not fail)
            #z = MM.dot(lastA,mat)
            z = MM.matmulS(lastA,mat)
            z_matrixs.append(z)
            a =activeFunc(z)
            l+=1
            if(l<layerSize):
                a = NeuralLeaf.XAddOnes(a)
            a_matrixs.append(a)
            lastA = a
        if success:
            return [Leaf(z_matrixs),Leaf(a_matrixs)]
    
    def backward(self,Z:'Leaf',A:'Leaf',DADzFunc,DJDzFunc,data)->Leaf:
        """
         w[l][0][j] :bias
        backward to get gradients dL/dwij  --> derivLW
        cached dL/dZ_k(l) cachedLZ
        L : LossFunc
        dL/dW_ij(l-1) = sum_k{ dL/dZ_k(l) * dZ_k(l)/dW_ij(l-1) }
        dL/dZ_k(l) = sum_k1{ dL/dZ_k1(l+1) * dZ_k1(l+1)/dZ_k(l) }
        dZ_k(l)/dw_ij(l-1) = a_i(if k==j, else =0)

        use matrix to calculate:
        DL/DZ(l) = DL/DZ(l+1) * W(l)^T * a'(l)
        DL/DW(l-1) = a(l-1)^T * DL/DZ(l)

        Z: value stored in network of matrix
        A: A = active(Z)
        Y: prepared correct result for X
        """
        #Log.Debug(NeuralLeafTag,"debug:backneurals begin")
        LayerSize = A.getLayerSize()
        cachedLZ = [None]*LayerSize
        l = LayerSize-1
        results = [None]*(LayerSize-1)
        while(l>0):
            al = A[l]
            # not last layer should remove addOne todo
            if (l!=(LayerSize-1)):
                al=A[l][:,1:]
            al_1 = A[l-1]
            DAl_DZl = DADzFunc(al,Z[l])
            if(l == LayerSize-1): # handle last layer of network todo
                cachedLZ[l] =  DJDzFunc(data.Y,al) # DJ/Dz(L) = DJ/Dy*DyDz(L)  y=a(L)
            else:
                dLdZlP1 = cachedLZ[l+1]
                cachedLZ[l] = MM.matmul(dLdZlP1,MM.transpose(self._matrixs[l][1:]))*DAl_DZl
            al_1_T = MM.transpose(al_1)
            #Log.Debug("TempTest_lZl",f"temp ones=\n{al_1_T[0]}")
            #Log.Debug("TempTest_lZl",f"cacheLZ l=\n{cachedLZ[l]}")
            #Log.Debug("TempTest_lZl",f"al_1_T l=\n{al_1_T[1:]}")
            #todo why would it save time? and make success
            results[l-1] = MM.matmulS(al_1_T,cachedLZ[l])
            # temp = np.ones([1,al.shape[0]])
            # check1 = np.matmul(temp,cachedLZ[l])
            #checkdjdb = MM.matmul(al_1_T[0]*1.0,cachedLZ[l])
            #checkdjdw12 = MM.matmul(al_1_T[1:],cachedLZ[l])
            # results[l-1][0] = checkdjdb
            # results[l-1][1] = MM.matmul(al_1_T[1:2],cachedLZ[l])
            # results[l-1][2] = MM.matmul(al_1_T[2:3],cachedLZ[l])

            # results[l-1][1] = checkdjdw12[0]
            # results[l-1][2] = checkdjdw12[1]
            #Log.Debug("TempTest_lZl",f"dJdb\n{results[l-1]},check1={checkdjdb},checkdjdw12={checkdjdw12} ")
            
            l-=1
        return Leaf(results)
    


    