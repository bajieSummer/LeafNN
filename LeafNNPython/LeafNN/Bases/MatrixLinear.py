import numpy as np
class MatrixLinear:
    def det(matA):
        return np.linalg.det(matA)
    
    def getInverse(matA):
        return np.linalg.inv(matA)
