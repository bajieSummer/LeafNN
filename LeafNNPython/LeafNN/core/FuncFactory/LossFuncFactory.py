import numpy as np
class LossFuncFactory:
    def BinaryClassify(Y, Y_p):
        """
        Y: prepared result for each example
        Y_p: predicted result for each example
        Binary classification: 
        L = -1*(y_i*log(y_p_i) + (1-y_i)*log(1-y_p_i)) 
        1. (i= 1 ~ n)
        2. y_i: the corrected result for ith example
        3. y_p_i: the predicted result for ith example
        1 first row of Y_p, Y means the first example, 2rd row relate to 2rd example
        """
        # not proper when training, leading to not converged
        # Epsilon = 1e-16
        # Y_p = np.clip(Y_p, Epsilon, 1.0 - Epsilon) 
        Loss = -1*( Y*np.log(Y_p)+(1.0-Y)*np.log(1.0-Y_p))
        Loss[((Y == 1.0)&(Y_p == 1.0)) |((Y==0.0)&(Y_p==0.0))] = 0.0 
        return Loss
    
    def DerivBinaryClassify(Y,Y_p):
        """
        Y: prepared result for each example
        Y_p: predicted result for each example
        dL/dY_p =-1*( y_i/y_p_i + (1-y_i)/(1-y_p_i) )
        """
        return (Y_p-Y)/(Y_p*(1-Y_p))
    
    # todo
    def DLDZBinaryClassify2Sigmoid(Y,Y_p):
        """
        Y: prepared result for each example
        Y_p: predicted result for each example
        DerivBinaryClassify dL/dy_p = -1*( y_i*1/y_p_i + (1-y_i)*1/(1-y_p_i) )
        DerivSigmoid : dy/dz = y*(1.0-y)    
        return the result of dL/dZ = dL/dY_p * dY_p/dZ = Y_p - Y
        """
        result = Y_p - Y
        return result
