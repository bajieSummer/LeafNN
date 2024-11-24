from LeafNN.utils.Log import Log
import LeafNN.core.LeafModels.ModelData as MD
from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.core.LeafModels.Leaf import Leaf
from LeafNN.ModelDataConverters.ConvertorFactory import ConvertorFactory
DataUtilsTag = "DataUtilsTag"
class DataUtils:
    def __init__(self):
        Log.Debug(DataUtilsTag,"DataUtils init")
    
    def Leaf2ClassifyData(leaf:Leaf):
        return MD.ClassifyData(leaf[0],leaf[1])
    
    def readDataXYFromFile(filePath,isTransposeX=False,picW = None)->MD.ClassifyData:
        leaf = ConvertorFactory.getInstance().readXYFromFile(filePath)
        data = DataUtils.Leaf2ClassifyData(leaf)
        if isTransposeX:
            [m,n] = data.X.shape
            pw = picW
            resX = MM.zeros(data.X.shape)
            if pw is None:
                pw = int(MM.sqrt(n))
            ph = int(n/pw)
                # todo pw*ph!=n warning or error
            for i in range(m):
                rxi = data.X[i,:].reshape([pw,ph])
                resX[i,:] = MM.transpose(rxi).flatten()
            data.X = resX
        return data

    def readWB(filePath,isTranspose=False):
        wb = ConvertorFactory.getInstance().readWB(filePath)
        if isTranspose:
            return wb.T()
        else:
            return wb
    
    def writeWB(leaf,filePath):
        return ConvertorFactory.getInstance().writeWB(leaf,filePath)
    
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
            if MM.isclose(min,0.0):
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


    def generateMeshPoints(X_origin,HighestPolyDegree,meshId0,meshId1,wb,predictFunc,isNormalLize=False,cross=True):
        """
        X_origin: X,HighestPolyDegree, the highest PolyDegree
        meshId0,meshId1, indexes of X_origin,  x_axis : X_origin[meshId0], y_axis:X_origin[meshId1]
        isNormalize: should X features scale to 0~1
        cross : X1*X1, X1*X2  ,X2*X2,  X1*X2 is the cross feature
        """
        [X_mins,X_maxs]=DataUtils.findFeatureMaxMin(X_origin)
        [m,n] = X_origin.shape
        Xres = None
        m2 =50
        for i in range(n):
            xfi = MM.linspace(X_mins[i], X_maxs[i], m2)[:,MM.newaxis]  # Range for x
            if Xres is None:
                Xres = xfi
            else:
                Xres = MM.hstack([Xres,xfi])
        # we create mesh based meshId0,meshId1 
        mY = MM.ones([m2,m2])
        for i in range(m2):
            xfid0i =Xres[:,meshId0][i]
            X_id0i = MM.ones([m2,1])*xfid0i
            mX = Xres*1.0
            mX[:,meshId0] = X_id0i.flatten()
            mX = DataUtils.preprocessData(mX,isNormalLize,HighestPolyDegree,cross)  
            mY[i,:] = predictFunc(mX,wb).flatten()

        return [Xres,mY]
        
            
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

            

            