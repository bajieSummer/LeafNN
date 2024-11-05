import numpy as np
from LeafNN.utils.Log import Log
from LeafNN.core.LeafModels.Leaf import Leaf
NeuralLeafTag = "NeuralLeafTag"
class NeuralLeaf(Leaf):
    def __init__(self,matrixArray):
        super().__init__(matrixArray)
    
    def XAddOnes(X):
        ones_x = np.ones([X.shape[0],1])
        X = np.hstack([ones_x,X])
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
            z = np.dot(lastA,mat)
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
        Log.Debug(NeuralLeafTag,"debug:backneurals begin")
        LayerSize = A.getLayerSize()
        cachedLZ = [None]*LayerSize
        l = LayerSize-1
        results = [None]*(LayerSize-1)
        while(l>0):
            al = A[l]
            al_1 = A[l-1]
            DAl_DZl = DADzFunc(al,Z[l])
            if(l == LayerSize-1): # handle last layer of network
                cachedLZ[l] =  DJDzFunc(data.Y,al) # DJ/Dz(L) = DJ/Dy*DyDz(L)  y=a(L)
            else:
                dLdZlP1 = cachedLZ(l+1)
                cachedLZ = np.matmul(dLdZlP1,np.transpose(self._matrixs[l]))*DAl_DZl
            al_1_T = np.transpose(al_1)
            results[l-1] = np.matmul(al_1_T,cachedLZ[l])
            l-=1
        return Leaf(results)
    


    