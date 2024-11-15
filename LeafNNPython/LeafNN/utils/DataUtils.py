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

    def getPolyX1X2(HightestPolyDegree,index1,index2,XPolys):
        i = 2
        res = None
        while (i <= HightestPolyDegree):
            j = 0
            while (j<=i):
                #X1^(i-j)*X2^j
                T = (XPolys[i-j][:,index1]*XPolys[j][:,index2])[:,MM.newaxis]
                if(res is None):
                    res = T
                else:
                    res = MM.hstack([res,T])
                j+=1
            i+=1
        return res




    # tod first normalize then multiPolyDegree?
    def preprocessData(X_input,isNormalize,HighestPolyDegree,cross=True):
        """
        isNormalize: should normalize the columns
        HighestPolyDegree: should >=1 should be integer
        cross: x1*x1 x1*x2 x2*x2, x1*x2 is the cross
        """ 
        X = X_input
        [m,n]=X_input.shape
        if isNormalize:
            X = DataUtils.normalizeColumn(X_input)
       
        Xmul = X
        resX = X
        XPolys = [MM.ones(X.shape),X]
        ip = 2
        while ip <= HighestPolyDegree:
            Xmul = Xmul*X
            XPolys.append(Xmul)
            ip+=1
        if cross:
            id1 = 0
            while id1 < n:
                id2=id1+1
                while id2 <n:
                    T = DataUtils.getPolyX1X2(HighestPolyDegree,id1,id2,XPolys)
                    resX = MM.hstack([resX,T])
                    id2+=1
                id1+=1
        else:
            i=2
            while i <= HighestPolyDegree:
                resX = MM.hstack([resX,XPolys[i]])
                i+=1
        #n features  x1 x2 x3
        # if Hpoly=2  x1,x2 x1*x2,x1*x1 x1*x2 x1*
        return resX

            

            