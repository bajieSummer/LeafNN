from LeafNN.utils.Log import Log
import numpy as np
import LeafNN.core.LeafModels.ModelData as MD
from LeafNN.Bases.MathMatrix import MathMatrix as MM
DataUtilsTag = "DataUtilsTag"
class DataUtils:
    def __init__(self):
        Log.Debug(DataUtilsTag,"DataUtils init")
    
    def readDataXYFromFile(filePath)->MD.ClassifyData:
        with open(filePath, 'r') as file:
            lineCount = 0
            data = []
            for line in file:
                elements = line.strip().split(',')  # Split the line by comma
                data.append([float(element) for element in elements])
                lineCount += 1
        # transform into np.array
        result = np.array(data)
        [n,m] = result.shape
        dataX = result[:,0:m-1]
        dataY = result[:,m-1:m]
        return MD.ClassifyData(dataX,dataY)
    
    def findFeatureMaxMin(X):
        (m,n) = X.shape
        featureMaxs = [-1.0*MM.inf()]*n
        featureMins = [MM.inf()]*n
        for x in X:
            for i in range(n):
                if featureMaxs[i] < x[i]:
                    featureMaxs[i] = x[i]
                if featureMins[i] > x[i]:
                    featureMins[i] = x[i]
        return [featureMins,featureMaxs]
    
    def getFrac(x,min,max):
        res = x
        length = max - min
        if(length == 0.0):
            if np.isclose(min,0.0):
                res = 0.0
            else:
                res = 1.0
        else:
            res = (x-min)/length
        return res
                
    def normalizeColumn(X):
        [featureMins,featureMaxs] = DataUtils.findFeatureMaxMin(X)
        i = 0
        resX = MM.ones(X.shape)
        for x in X:
            j = 0
            for xi in x:
                min = featureMins[j]
                max = featureMaxs[j]
                resX[i][j] = DataUtils.getFrac(xi,min,max)
                j+=1
            i+=1
        return resX

    # tod first normalize then multiPolyDegree?
    """
    isNormalize: should normalize the columns
    HighestPolyDegree: should >=1 should be integer
    """  
    def preprocessData(X_input,isNormalize,HighestPolyDegree):
        X = MM.ones(X_input.shape)
        if isNormalize:
            X = DataUtils.normalizeColumn(X_input)
        i = 1
        Xmul = X
        resX = X
        while i < HighestPolyDegree:
            Xmul = Xmul*X
            resX = MM.hstack([resX,Xmul])
            i+=1
        return resX

            

            