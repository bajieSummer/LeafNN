import numpy as np
class ModelEvaluation:
    def getBinaryClassifyAccuracy(Y,Y_p):
         #todo only for binary classification
        n = len(Y)
        if(n == 0):
            print("the test data set is empty")
            return -1
        passNums = 0
        for i in range(n):
            if(np.abs(Y[i]-Y_p[i]) < 0.5): # for sigmoid func and classification loss func (todo)
                passNums +=1
        return passNums/n