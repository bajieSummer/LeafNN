from LeafNN.core.Bases.MathMatrix import MathMatrix as MM
#import numpy as np
class ActiveFuncFactory:
    def Sigmoid(X):
        """
        sigmoid func
        input: inputX 
        """
        return 1.0/(1.0+MM.exp(-X))
    
    def DerivSigmoid(X):
        """
        derivative of sigmoid func  d_sigmoid = sigmoid*(1-sigmoid)
        ds/dx = e^(-x)/(1+e^(-x))^2 = 1/(1+e^(-x))*(1 -1/(1+e^(-x))
        X 
        """
        sig = 1.0/(1.0+MM.exp(-X))
        return sig*(1.0-sig)
    
    def DerivSigmoidFromS(sigmoidValue):
        """
        derivative of sigmoid func  d_sigmoid = sigmoid*(1-sigmoid)
        ds/dx = e^(-x)/(1+e^(-x))^2 = 1/(1+e^(-x))*(1 -1/(1+e^(-x)) 
        ds/dx = s*(1-s) = sigmoidValue*(1.0-sigmoidValue)
        """
        return sigmoidValue*(1.0-sigmoidValue)
