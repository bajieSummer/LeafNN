from LeafNN.Bases.MathMatrix import MathMatrix as MM
# import numpy as np
from LeafNN.core.LeafModels.Leaf import Leaf
from LeafNN.core.LeafModels.NeuralLeaf import NeuralLeaf
from LeafNN.core.FuncFactory.ActiveFuncFactory import ActiveFuncFactory as ActiveF
from LeafNN.core.FuncFactory.LossFuncFactory import LossFuncFactory as LossF
from LeafNN.utils.Log import Log
from LeafNN.core.LeafModels.TrainOptions import TrainOption as TOps
from LeafNN.core.FuncFactory.OptimalFuncFactory import OptimalFuncFactory as OptFG
import LeafNN.core.LeafModels.TrainMonitor  as TMot
from LeafNN.core.LeafModels.ModelData import ClassifyData

modelTag = 'BaseClassifyModel'
class BaseClassifyModel:
    # wb matrix array

    def layerCheck(wb:NeuralLeaf,layerNodeSizeList):
        isSuccess = True
        if(len(layerNodeSizeList)!=wb.getLayerSize()+1):
            Log.Error(modelTag,"layerCheck failed, the wb+1 dosen't have the same layers as layerNodeSizeList")
            isSuccess = False
        for i in range(wb.getLayerSize()):
            (m,n)=wb[i].shape
            if(m!=(layerNodeSizeList[i]+1) or n!=layerNodeSizeList[i+1]):
                Log.Error(modelTag,f"wb shape doesn't match layerNodeSizeList i={i},wb_shape={wb[i].shape},layerNodeSizeList={layerNodeSizeList}")
                isSuccess=False
            i+=1
        if isSuccess:
            return True
        else:
            return False
        
    def __init__(self,layerNodeSizeList,wb:NeuralLeaf=None,trainOption:TOps=None):
        self.layerNodeSizeList = layerNodeSizeList
        self.trainOption = trainOption
        if(self.trainOption is None):
            self.trainOption = TOps()
        if(wb is None):
            self.wb = self.randInitializeWb(layerNodeSizeList)
        else:
            if(not BaseClassifyModel.layerCheck(wb,layerNodeSizeList)):
                Log.Error(modelTag,"init Failed")
                self.wb = None
            else:
                self.wb = wb
        self.trainData = None
        self.testData = None
        self.validateData = None
        self.grads = None
        self.activeFunc  = None
        self.derivActiveFunc = None
        self.lossFunc = None
        self.DerivLossFunc = None
       
    def randInitializeWb(self,layerNodeSizeList):
        Log.Debug(modelTag,f"init>>")
        # 1. initial weights and bias
        # l = 0
        # self.modelWeights = [None]*(self.layerSize-1)
        # self.modelBias = [None]*(self.layerSize-1)
        # while(l < self.layerSize - 1):
        #     self.modelWeights[l] = np.random.rand(self.layerNodeSizeList[l],self.layerNodeSizeList[l+1])
        #     self.modelBias[l] = np.random.rand(1,self.layerNodeSizeList[l+1])
        #     l+=1
        layerSize = len(layerNodeSizeList)
        WBmats = [None]*(layerSize-1)
        l = 0
        while(l < layerSize - 1):
            # self.layerNodeSizeList[l]+1, +1 for bias
            WBmats[l] = MM.rand(self.layerNodeSizeList[l]+1,self.layerNodeSizeList[l+1])*2.0-1.0
            l+=1
        return NeuralLeaf(WBmats)

    def setData(self,data:ClassifyData):
        if(data.X.shape[0]!=data.Y.shape[0]):
            Log.Error(modelTag,"dataX and dataY don't have the same size")
            return
        if(data.X.shape[1]!=self.layerNodeSizeList[0]):
            Log.Error(modelTag,"the input data should have the same features as the size of self.layerNodeSizeList[0]")
        #set data 
        n = data.X.shape[0]
        trainIndsEnd = int(n*(self.trainOption.trainRatio))
        testIndsStart =int((1.0 - self.trainOption.testRatio)*n)

        trainX = data.X[0:trainIndsEnd,:]
        trainY = data.Y[0:trainIndsEnd,:]
        validX = data.X[trainIndsEnd:testIndsStart,:]
        validY = data.Y[trainIndsEnd:testIndsStart,:]
        testX = data.X[testIndsStart:n,:]
        testY = data.Y[testIndsStart:n,:]
        self.trainData = ClassifyData(trainX,trainY)
        self.validateData = ClassifyData(validX,validY)
        self.testData = ClassifyData(testX,testY)

        
        #check Length
    
    def setActiveFunc(self,activeFunc):
        self.activeFunc = activeFunc
    
    def active(self,X):
        if(self.activeFunc is None):
            self.activeFunc = ActiveF.Sigmoid
        return self.activeFunc(X)

    def setDerivActiveFunc(self,derivActiveFunc):
        self.derivActiveFunc = derivActiveFunc
    
    """
    a=sigmoid(z)
    da/dz = a*(1-a)
    """
    def derivActive(self,a,z):
        if(self.derivActiveFunc is None):
            return ActiveF.DerivSigmoidFromS(a)
        else:
            return self.derivActiveFunc(a,z)
    
    def setLosssFunc(self,lossFunc):
        self.lossFunc = lossFunc
    
    def Loss(self,Y,Y_predict):
        if(self.lossFunc is None ):
            return LossF.BinaryClassify(Y,Y_predict)
        return self.lossFunc(Y,Y_predict)
    
    def setDerivLossFunc(self,DLDzFunc):
        self.DerivLossFunc = DLDzFunc

    def DerivLoss(self,Y,Y_predict):
        if(self.DerivLossFunc is None):
            return LossF.DLDZBinaryClassify2Sigmoid(Y,Y_predict)
        return self.DerivLossFunc(Y,Y_predict)# todo got problem,dLdz might need Z
    
    def __DJDy(self,Y,Y_predict):
        n = Y_predict.shape[0]
        return 1/n*self.DerivLoss(Y,Y_predict)

    def predictWithCache(self,X,wb:NeuralLeaf):
        [Z,A]=wb.forward(X,self.active)
        return [Z,A]
    
    def predict(self,X,wb:NeuralLeaf):
        if(wb is None or X is None):
            Log.Error(modelTag,"parameters can't be none (wb,X)")
            return None
        [Z,A]=self.predictWithCache(X,wb)
        return A[A.getLayerSize()-1]
    
    def __calCostFromY(self,wb:NeuralLeaf,Y,Y_predict):
        n = len(Y_predict)
        if(n>0):
            cost = 1.0/n*MM.sum(self.Loss(Y,Y_predict))
            if self.trainOption.regularEnable:
                cost += self.trainOption.regularLamada/(2*n)*(wb*wb)
                # why we should minus bias*bias?
                biasM = 0
                for i in range(wb.getLayerSize()):
                    biasM+=MM.sum(wb[i][0]*wb[i][0])
                cost -=self.trainOption.regularLamada/(2*n)*biasM
        else:
            cost = 0
        
        return cost
    
    def calCost(self,wb:NeuralLeaf,dataXY:ClassifyData)->float:
        if(wb is None or dataXY is None):
            Log.Error(modelTag,"parameters can't be none (wb,dataXY)")
            return None
        outputY= self.predict(dataXY.X,wb)
        cost = self.__calCostFromY(wb,dataXY.Y,outputY)
        return cost

    def __calGradsFromCache(self,wb:NeuralLeaf,dataXY:ClassifyData,Z,A):
        grads = wb.backward(Z,A,self.derivActive,self.__DJDy,dataXY)
        if self.trainOption.regularEnable:
            m = len(dataXY.Y)
            gradReg = self.trainOption.regularLamada*1/m*wb
            # why regularization doesn't consider bias?
            for i in range(wb.getLayerSize()):
                n = len(gradReg[i][0])
                for j in range(n):
                    gradReg[i][0][j] = 0
            grads +=gradReg
        return grads  
              
    def calGrads(self,wb:NeuralLeaf,dataXY:ClassifyData)->Leaf:
        [Z,A] = self.predictWithCache(dataXY.X,wb)
        grads = self.__calGradsFromCache(wb,dataXY,Z,A)
        return grads

  
    def calCostAndGrads(self,wb:NeuralLeaf,dataXY:ClassifyData):
        """
        return [cost,grads] : type[float,Leaf]
        """
        if(wb is None or dataXY is None):
            Log.Error(modelTag,"parameters can't be none (wb,dataXY)")
            return None
        [Z,A] = self.predictWithCache(dataXY.X,wb)
        outputY = A[A.getLayerSize()-1]
        cost = self.__calCostFromY(wb,dataXY.Y,outputY)
        grads = self.__calGradsFromCache(wb,dataXY,Z,A)
        return[cost,grads]

    def train(self,monitorOption:TMot.MonitorOption=None,outMonitorData:TMot.MonitorData=None):
        # training
        Log.Debug(modelTag,f"train begin>>")
        if(self.trainData is None):
            Log.Error(modelTag,"invalid context, there is no train data setted")
            return False  
        if self.wb is None: 
            Log.Error(modelTag,"train failed wb is None")
            return False
        [self.wb,mData] = OptFG.OptimalMinWithWolfes(self.calCostAndGrads,self.wb,self.trainOption,monitorOption,self.trainData)
        if(monitorOption and monitorOption.enable):
            TMot.TrainMonitor.monitor(monitorOption,outMonitorData,mData.costs,mData.rates,mData.grads)
            Log.Debug(modelTag,f"final train cost={outMonitorData.costs}")
            Log.Debug(modelTag,f"final train rates={outMonitorData.rates}")
        return True

        
        

    


        

        

    

