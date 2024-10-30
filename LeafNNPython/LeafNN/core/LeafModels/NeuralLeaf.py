import numpy as np
from LeafNN.core.LeafModels.Leaf import Leaf
class NeuralLeaf(Leaf):
    def __init__(self,matrixArray):
        super.__init__(matrixArray)
    
    def XAddOnes(X):
        ones_x = np.ones([X.shape[0],1])
        X = np.hstack([ones_x,X])
        return X
    
    def forward(self,X_input,activeFunc):
        #WX
        X = NeuralLeaf.XAddOnes(X_input)
        z_matrixs = [None]
        a_matrixs = [X]
        success = True
        lastA = X
        for mat in self.__matrixs:
            z = np.dot(lastA,mat)
            z = NeuralLeaf.XAddOnes(z)
            z_matrixs.append(z)
            a_matrixs.append(activeFunc(z))
            lastA = z
        if success:
            return [Leaf(z_matrixs),Leaf(a_matrixs)]
    
    def backward(self,Z:'Leaf',A:'Leaf',DADzFunc,DJDzFunc,data):
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
        print("debug:backneurals begin")
        LayerSize = A.getLayerSize()+1
        cachedLZ = [None]*self.layerSize
        l = LayerSize-1
        results = [None]*LayerSize
        while(l>0):
            al = A[l]
            al_1 = A[l-1]
            DAl_DZl = DADzFunc(al,Z[l])
            if(l == LayerSize-1): # handle last layer of network
                cachedLZ[l] =  DJDzFunc(al) # DJ/Dz(L) = DJ/Dy*DyDz(L)  y=a(L)
            else:
                dLdZlP1 = cachedLZ(l+1)
                cachedLZ = np.matmul(dLdZlP1,np.transpose(self.__matrixs[l]))*DAl_DZl
            al_1_T = np.transpose(al_1)
            results[l-1] = np.matmul(al_1_T,cachedLZ[l])
            l-=1
        return Leaf(results)
    


    