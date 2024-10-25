'''
Author: Sophie
email: bajie615@126.com
Date: 2024-07-10 14:09:06
Description: file content
'''
import copy
import numpy as np
import LeafNN.core.DLModels.TrainOptions as tp
from LeafNN.core.DLModels.BaseModel import BaseModel
from LeafNN.utils.MathUtils import MathUtils 
from LeafNN.core.DLModels.GradientDescentFactory import GradientDescentFactory as GF
from LeafNN.core.DLModels.ModelEvaluation import ModelEvaluation as ME
from LeafNN.core.FuncFactory.ActiveFuncFactory import ActiveFuncFactory as ActiveF

class SimpleFCModel(BaseModel):
    def __init__(self,layerSize,layerNodeSizeList):
        super(SimpleFCModel,self).__init__(layerSize,layerNodeSizeList)
        #train Option: trainInputData: X,Y
        self.trainX = None
        self.trainY = None
        #train option: cache gradients
        self.derivLW = [None]*(self.layerSize-1)
        self.derivLB = [None]*(self.layerSize-1)
        #train option: learning rate and maxIteraionNumber when training
        self.learnRate = 0.001
        self.maxIterationNum = 1000 
        #train option: early stop option
        self.enableEarlyStop = True
        self.__ETLastCost = 0
        self.ETFrequency = 100
        self.ETminiDiffCost = 0.01
        #train option: gradient check option
        self.enableGradientCheck = True 
        self.gradientCheckFrequency = 100
        #train option: quickDerivDlDz
        self.quickDerivDlDz = None
        # train option: initWeights:
        self.initWeights = None

        #test option: 
        self.testX = None
        self.testY = None
        # for binary classification it means abs(y-y_p)<0.5 pass otherwise fail
        self.grossTestPassRate = None 
        #model Option: train/cross validation/ test
        self.trainProportion = 0.7
        self.crossValidationProportion = 0.0
        self.testProportion = 0.3
    
    def defaultCost(Y, Y_p):
        """
        Binary classification: 
        L = -1/n * sum(y_i*log(y_p_i) + (1-y_i)*log(1-y_p_i)) 
        1. (i= 1 ~ n)
        2. y_i: the corrected result for ith example
        3. y_p_i: the predicted result for ith example
        1 first row of Y_p, Y means the first example, 2rd row relate to 2rd example
        """
        Epsilon = 1e-15
        n = len(Y)
        Cost = None
        #Y_p = np.clip(Y_p, Epsilon, 1.0 - Epsilon) 
        T = Y*np.log(Y_p)+(1.0-Y)*np.log(1.0-Y_p)
        # to avoid nan
        T[((Y == 1.0)&(Y_p == 1.0)) |((Y==0.0)&(Y_p==0.0))] = 0.0  # Loss is zero when both Y and Y_p are 1
        Cost = -1.0/n*np.sum(T,axis=0)
        Cost = Cost.item()
        # if(np.isnan(Cost.any())):
        #     print(Y_p)
        #     Cost = -1.0/n*np.sum(Y*np.log(Y_p+Epsilon)+(1.0-Y)*np.log(1.0-Y_p+Epsilon),axis=0)
        #     print("here is nan")
        if np.isnan(Cost):
            print("here1")
        if np.isinf(Cost):
            print("here2")
        return Cost
    
    def DerivDefaultLoss(Y,Y_p):
        """
        dL/dY_p = -1/n * sum (y_i*1/y_p_i + (1-y_i)*1/(1-y_p_i))
        """
        n = len(Y)
        return -1/n*np.sum((Y+Y_p-2*Y*Y_p)/(Y_p*(1-Y_p)),axis=0)
    
    def __quickDefaultDLDYMainFunc(Y,Y_p):
        """
        default Loss L = -1/n * sum(y_i*log(y_p_i) + (1-y_i)*log(1-y_p_i)) 
        L = sum(T(y,y_p))
        T(y,y_p) = -1/n*(y*log(y_p) + (1-y)*log(1-y_p) )
        DL/DY_P = sum(DT/DZ)
        return the result of DT/DY_P
        """
        n = len(Y)
        result = -1/n*(Y-Y_p)
        return result
     
    def hasInitialized(self):
        if self.modelBias is None or self.modelWeights is None:
            return False
        return True
    
    def __activeNode(self,input):
        if(self.activeFunc is None):
            return ActiveF.Sigmoid(input)
        else:
            return self.activeFunc(input)
        
    def __calCost(self,outputY):
        if(self.lossFunc is None ):
            return SimpleFCModel.defaultCost(self.trainY,outputY)
        else:
            n = len(outputY)
            return 1/n*np.sum(self.lossFunc(self.trainY,outputY))
    
    def __derivDlDy(self,outputY):
        if(self.quickDerivDlDz is None):
            return SimpleFCModel.__quickDefaultDLDYMainFunc(self.trainY,outputY)
        else:
            return self.quickDerivDlDz(self.trainY,outputY)
    def __derivDADz(self,a,z):
        if(self.derivActiveFunc is None):
            return ActiveF.DerivSigmoidFromS(a)
        else:
            return self.derivActiveFunc(z)
    def predictWithParams(self,inputX,weights,bias):
        """
        inputX : inputData for the Model
        the inputX.shape =(w,z) w:how many examples of input, z:how many nodes of input 
        eg1: [[255,233,211]], shape:1x3, which means the inputX got 3 nodes, only 1 example
        eg2: [[255,233,211],[[255,233,211]]]  shape:2x3, which means the inputX got 3 nodes, got 2 example
        """
        print("predictWithParams begin")
        if not self.hasInitialized():
            print("weights,bias not initialized")
            return 
        if(not isinstance(inputX,np.ndarray)):
            print("invalid inputX, inputX is not a tuple")
            return
        if len(inputX.shape)!=2 :
            print(f" invalid inputX shape, the inputX should be handled into 2 dimension.")
            return
        if inputX.shape[1]!=self.layerNodeSizeList[0]:
            print(f"invalid inputX,incosistent input, the Model layerNodeSizeList define the inputData should have {self.layerNodeSizeList[0]} nodes,but inputX got {inputX.shape[1]} nodes")
        # outputY, cacheA, cacheZ.
        # Z(l+1) = Z(l)*W(l)+b(l)
        # A(l) = active(Z(l))
        # Z(0) = inputX
        cacheA = [None]*self.layerSize
        cacheZ = [None]*self.layerSize
        cacheZ[0] = np.zeros(inputX.shape)
        cacheA[0] = inputX
        print("debug:inputX shape is", inputX.shape)
        print("debug:cacheA[0] shape",cacheA[0].shape)
        print("debug:weights[0] shape is ",weights[0].shape)
        print("debug:bias[0] shape is",bias[0].shape)
        layer = 1
        while layer < self.layerSize :
            cacheZ[layer] = np.matmul(cacheA[layer-1], weights[layer-1]) + bias[layer-1]
            cacheA[layer] = self.__activeNode(cacheZ[layer])
            layer +=1
        outputY = cacheA[self.layerSize-1]
        # print("predicted outputY")
        # print(outputY)
        print("predictWithParams end")
        return outputY,cacheA,cacheZ


    def predict(self,inputX):
        return self.predictWithParams(inputX,self.modelWeights,self.modelBias)

    def setData(self,dataX,dataY):
        """
        dataX: a matrix of X. input data
          eg: dataX([[x1a,x1b,x1c],[x2a,x2b,x2c]])
          dataX got 2 examples(x1,x2), and 3 features(a,b,c) which got 3 nodes for input layer of network
          each row represent an example.
        dataY: a matrix of Y, prepared for classification
          eg: dataY([[Y1i],[Y2i],[Y3i]])
          dataY prepared classification results for 3 examples, and got 1 feature
        """
        if(dataX is None or dataY is None):
            print("dataX and dataY should not be None")
            return
        if(not isinstance(dataX,np.ndarray)):
            print("dataX should be an numpy adarray")
            return
        if(not isinstance(dataY,np.ndarray)):
            print("dataY should be an numpy adarray")
            return
        n = len(dataX)
        if(n != len(dataY)):
            print("Input data wrong: the dataX and dataY should have the same length (same examples)")
            return
        trainIndsEnd = int(n*(self.trainProportion))
        testIndsStart =int((1.0 - self.testProportion)*n)
        self.trainX = dataX[0:trainIndsEnd,:]
        self.trainY = dataY[0:trainIndsEnd,:]
        self.testX = dataX[testIndsStart:n,:]
        self.testY = dataY[testIndsStart:n,:]
        

    def __initTrain(self):
        if(self.initWeights!=None):
            (self.modelWeights,self.modelBias) = self.initWeights()
            return
        # 1. initial weights and bias
        l = 0
        self.modelWeights = [None]*(self.layerSize-1)
        self.modelBias = [None]*(self.layerSize-1)
        while(l < self.layerSize - 1):
            self.modelWeights[l] = np.random.rand(self.layerNodeSizeList[l],self.layerNodeSizeList[l+1])
            self.modelBias[l] = np.random.rand(1,self.layerNodeSizeList[l+1])
            l+=1

    def __backwardNeurals(self,cacheA,cacheZ):
        """
        backward to get gradients dL/dwij, dL/dbj  --> derivLW, derivLB
        cached dL/dZ_k(l) cachedLZ
        L : LossFunc
        dL/dW_ij(l-1) = sum_k{ dL/dZ_k(l) * dZ_k(l)/dW_ij(l-1) }
        dL/dZ_k(l) = sum_k1{ dL/dZ_k1(l+1) * dZ_k1(l+1)/dZ_k(l) }
        dZ_k(l)/dw_ij(l-1) = a_i(if k==j, else =0)
        dZ_k(l)/db_j(l-1) = 1(if k==j, else =0)

        use matrix to calculate:
        DL/DZ(l) = DL/DZ(l+1) * W(l)^T * a'(l)
        DL/DW(l-1) = a(l-1)^T * DL/DZ(l)
        DL/DB(l-1) = ones_a(l-1)^T * DL/DZ(l)

        cacheA: value stored in network
        cacheZ: A = active(Z)
        Y: prepared correct result for X
        """
        print("debug:backneurals begin")
        cachedLZ = [None]*self.layerSize
        l = self.layerSize-1
        while(l>0):
            al = cacheA[l]
            al_1 = cacheA[l-1]
            DAl_DZl = self.__derivDADz(al,cacheZ[l])
            if(l == self.layerSize-1): # handle last layer of network
                cachedLZ[l] =  self.__derivDlDy(al)
            else:
                dLdZlP1 = cachedLZ(l+1)
                cachedLZ = np.matmul(dLdZlP1,np.transpose(self.modelWeights[l]))*DAl_DZl
            al_1_T = np.transpose(al_1)
            self.derivLW[l-1] = np.matmul(al_1_T,cachedLZ[l])
            self.derivLB[l-1] =  np.matmul(np.ones([1,al.shape[0]]),cachedLZ[l])
            l-=1
        print("debug:backneurals end")
  
    def __normalizedGradients(self,grads,l):
        grads_norm = np.linalg.norm(grads[l], axis=0, keepdims=True)
        # Define a constant for epsilon to prevent division by zero
        epsilon = 1e-7
        # Normalize the gradients along the specified axis
        grads[l] = grads[l] / (grads_norm + epsilon)


    def updateWithGradients(self):
        print("debug:updateWithGradients")
        for l in range(self.layerSize-1):
            print(f"debug:l={l},dldw_shape={self.derivLW[l].shape},weightsShape={self.modelWeights[l].shape}")
            print(f"debug: l={l},derivB shape={self.derivLB[l].shape}")
            #self.__normalizedGradients(self.derivLW,l)
            self.modelWeights[l] += -1.0*self.learnRate*self.derivLW[l]
            self.modelBias[l] += -1.0*self.learnRate*self.derivLB[l] 
            print(f"l={l} derivLW>>>")
            print(self.derivLW[l])
            print(f"l={l} derivLB>>>")
            print(self.derivLB[l])
            print(f"l={l},weight>>")
            print(self.modelWeights[l])
            print(f"l={l},bias>>")
            print(self.modelBias[l])


    def __isEarlyStop(self,outputY,iterationNum):
        result = False
        if(iterationNum!=0 and iterationNum%self.ETFrequency == 0):
            currentCost = self.__calCost(outputY)
            # todo : gradients/cost might vibrate if learning rate is too large to fit.
            diffL = currentCost-self.__ETLastCost
            if(np.abs(diffL)<=self.ETminiDiffCost):
                result = True
            if(diffL<0):
                print("cost vibration when training")
            self.__ETLastCost = currentCost
        return result
    
    def __checkGradients(self,iterationNum):
        print(f"check_gradients begin>>iterationNum={iterationNum}")
        if(iterationNum%self.gradientCheckFrequency !=0):
            return True
        print("check_gradients22>>,modelWeights")
        # wn1 = w_ij(l)+ ep1 
        # wn2 = w_ij(l)- ep1
        cacheWn1 = copy.deepcopy(self.modelWeights)
        print(cacheWn1)
        cacheCheckDlDW = [None]*(self.layerSize-1) 
        # bn1 = b_j(l)+ ep1 
        # bn2 = b_j(l)- ep1
        cachebn1 = copy.deepcopy(self.modelBias)
        print(cachebn1)
        cacheCheckDlDB = [None]*(self.layerSize-1)
        isCheckPass = True 
        for l in range(self.layerSize-1):
            nl1 = self.layerNodeSizeList[l]
            nl2 = self.layerNodeSizeList[l+1]
            cacheCheckDlDW[l] = np.zeros([nl1,nl2])
            cacheCheckDlDB[l] = np.zeros([1,nl2])
            print(f"checkgradients_in layer{l},l nodes={nl1}, l+1 nodes={nl2},dldB shape={self.derivLB[l].shape}")
            for i in range(nl1):
                for j in range(nl2):
                    # wn1 = wij+epsilon
                    print(f"checkgradients_dldw l={l},i={i},j={j} dldw={self.derivLW[l][i][j]}")
                    eps = MathUtils.Epsilon
                    cacheWn1[l][i][j] += eps
                    print("predict Y1(wi+epsilon)>>")
                    outputY1,cacheA,cacheZ = self.predictWithParams(self.trainX,cacheWn1,self.modelBias)
                    print(f"i ={i},j={j} z(L)={cacheZ[self.layerSize-1]} A(L)={cacheA[self.layerSize-1]}")
                    J1 = self.__calCost(outputY1)
                    print("cost>>",J1)
                    # wn2 = wij - epsion
                    cacheWn1[l][i][j] -= 2*eps
                    print("predict Y2(wi-epsilon)>>")
                    outputY2,cacheA,cacheZ = self.predictWithParams(self.trainX,cacheWn1,self.modelBias)
                    print(f"i ={i},j={j} z(L)={cacheZ[self.layerSize-1]} A(L)={cacheA[self.layerSize-1]}")
                    J2 = self.__calCost(outputY2)
                    chekdW = J1/(2.0*eps) - J2/(2.0*eps)
                    print(f"checkgradients_dldw l={l} i={i},j={j} cost1>{J1} cost2>{J2},checkdldw={chekdW},computerdldw = {self.derivLW[l][i][j]}")
                    cacheCheckDlDW[l][i][j]= chekdW 
                    if(np.abs(self.derivLW[l][i][j] - chekdW)>MathUtils.GradCheckMinDiff):
                        print(f"gradient checking: computing_dldw_wrong : checkDiff chekdW={chekdW},computerdldw = {self.derivLW[l][i][j]},layer={l},i={i},j={j}:")
                        isCheckPass = False
                    #recover wij
                    cacheWn1[l][i][j] += eps
                    #computer b
                    if( i == 0):
                        cachebn1[l][i][j] += eps
                        outputY1,cacheA,cacheZ = self.predictWithParams(self.trainX,self.modelWeights,cachebn1)
                        J1 = self.__calCost(outputY1)
                        print(f"b={cachebn1[l][i][j]}")
                        cachebn1[l][i][j] -= 2*eps
                        print(f"b={cachebn1[l][i][j]}")
                        outputY2,cacheA,cacheZ = self.predictWithParams(self.trainX,self.modelWeights,cachebn1)
                        J2 = self.__calCost(outputY2)
                        checkDB =  (J1-J2)/(2*eps)
                        cachebn1[l][i][j] += eps
                        cacheCheckDlDB[l][i][j] = checkDB
                        print(f"checkgradients_dldb l={l},i={i},j={j} dldb={self.derivLB[l][i][j]} checkDB={checkDB} j1={J1},j2={J2}")
                        if(np.abs(self.derivLB[l][i][j] - checkDB)>MathUtils.GradCheckMinDiff):
                            print(f"gradient checking: computing_dldb_wrong : checkDiff diffDB={checkDB}: computer dldb:{self.derivLB[l][i][j]} layer={l},i={i},j={j} diff={self.derivLB[l][i][j] - checkDB}")
                            isCheckPass = False
        if not isCheckPass:
            print(f"Check Not Passed: the differences between check result and gradient computing:")
        print(f"check_gradients end>>>>>>")
        return isCheckPass
    
    def __monitorTrainWithJ(self,depth,J,grads,monitorOption,resultMonitorData):
        if(monitorOption == None):
            return
        if(not monitorOption.enable):
            return
        if(resultMonitorData == None):
            print("not inital MonitorData")
            return
        print("begin to monitor training process")
        if(depth%(monitorOption.frequency)==0):
            print(f"monitor_train iterationNum={depth} cost={J}")
            resultMonitorData.iterationInds.append(depth)
            resultMonitorData.costs.append(J)
            resultMonitorData.grads.append(grads)
        
    def __mointor(self,monitorOption,resultMonitorData,costs,rates,grads):
        if(monitorOption and not monitorOption.enable):
            return 
        if(resultMonitorData == None):
            print("not inital MonitorData")
            return
        resultMonitorData.costs =copy.deepcopy(costs)
        resultMonitorData.rates = copy.deepcopy(rates)
        resultMonitorData.grads = copy.deepcopy(grads)
        for i in range(len(costs)):
            resultMonitorData.iterationInds.append(i)

        
    def __mointorTrain(self,depth,outputY,monitorOption,resultMonitorData):
        if(monitorOption == None):
            return
        if(not monitorOption.enable):
            return
        if(resultMonitorData == None):
            print("not inital MonitorData")
            return
        print("begin to monitor training process")
        if(depth%(monitorOption.frequency)==0):
            cost = self.__calCost(outputY)
            curDw = copy.deepcopy(self.derivLW)
            curDb = copy.deepcopy(self.derivLB)
            grad = (curDw,curDb)
            print(f"monitor_train iterationNum={depth} cost={cost}")
            resultMonitorData.iterationInds.append(depth)
            resultMonitorData.costs.append(cost)
            resultMonitorData.grads.append(grad)

    def __calCostWithParams(self,wb):
        [outputY,cacheA,cacheZ] =self.predictWithParams(self.trainX,wb[0],wb[1])
        cost = self.__calCost(outputY)
        return [cost,cacheA,cacheZ]
    
    def __calGradsWithCache(self,cacheA,cacheZ):
        self.__backwardNeurals(cacheA,cacheZ)
        return copy.deepcopy([self.derivLW,self.derivLB])

   
    def train(self,monitorOption=None,outMonitorData=None):
        """
        simple version of training prototype
        train function:
        Make sure have set data already: self.setData
        """
        print('train begin>>>')
        trainCompleted = False
        # 1. initial weights
        # 2. use weights to go forward predict.
        # 3. backward to get gradients
        # 4. update weights use some certain gradient method :eg: wi = wi - alpha*dL/dwi 
        # 5. go to step2, till converge.
        # 6. converge condition: dL/dwi reach near zero, succeed.
        # 7. failed: reach the maximum iteration 
        if(self.trainX is None or self.trainY is None):
            print("invalid context, there is no train data setted")
            trainCompleted = False
            return trainCompleted
        self.__initTrain()
        depth = 0
        while (depth < self.maxIterationNum):
            # use current new weights to predict and calculate nodes (cacheA,cacheZ)
            print("train iteration: depth=",depth)
            outputY,cacheA,cacheZ = self.predict(self.trainX)
            if(depth == 0):
                print(f"train_start iterationNum={depth} cost={self.__calCost(outputY)}")  
            self.__backwardNeurals(cacheA,cacheZ)
            if(self.enableEarlyStop and self.__isEarlyStop(outputY,depth)):
                break # earlyStop
            if(self.enableGradientCheck and not self.__checkGradients(depth)):
                trainCompleted = False
                break # gradientCheck
            self.__mointorTrain(depth,outputY,monitorOption,outMonitorData)
            self.updateWithGradients()
            trainCompleted = True
            depth+=1
       
        print('train end>>>')
        return trainCompleted

    def train2(self,monitorOption=None,outMonitorData=None):
        """
        train function:
        Make sure have set data already: self.setData
        """
        print('train begin>>>')
        trainCompleted = False
        # 1. initial weights
        # 2. use weights to go forward predict.
        # 3. backward to get gradients
        # 4. update weights use some certain gradient method :eg: wi = wi - alpha*dL/dwi 
        # 5. go to step2, till converge.
        # 6. converge condition: dL/dwi reach near zero, succeed.
        # 7. failed: reach the maximum iteration 
        if(self.trainX is None or self.trainY is None):
            print("invalid context, there is no train data setted")
            trainCompleted = False
            return trainCompleted
        self.__initTrain()
        depth = 0
        wb = [self.modelWeights,self.modelBias]
        [J1,cacheA,cacheZ] = self.__calCostWithParams(wb)
        grads = self.__calGradsWithCache(cacheA,cacheZ)
        lastGrads = copy.deepcopy(grads)
        wb_new = []
        print(f"train2 >> J1={J1},grads dldw={grads[0][0]},grads dldb={grads[1][0]}")
        while (depth < self.maxIterationNum):
            # Y = cacheA[self.layerSize-1]
            # # use current new weights to predict and calculate nodes (cacheA,cacheZ)
            # if(self.enableEarlyStop):
            #     if(self.__isEarlyStop(Y,depth)):
            #         break # earlyStop
            if(self.enableGradientCheck and not self.__checkGradients(depth)):
                trainCompleted = False
                break # gradientCheck
            self.__monitorTrainWithJ(depth,J1,grads,monitorOption,outMonitorData)
            print(f"beforeBatchGradientUpdate>>>depth={depth}cost={J1},weight={wb[0][0]} bias={wb[1][0]}")
            
            [wb_new,grad2,J2,alpha] = GF.BatchGradientWithLineSearch(wb,self.__calCostWithParams,self.__calGradsWithCache,J1,grads,depth,lastGrads)
            print(f"afterBatchGradientUpdate>>>cost={J2},alpha={alpha},weight={wb_new[0][0]} bias={wb_new[1][0]}")
            J1 = J2
            lastGrads = grads
            grads = grad2
            wb = wb_new
            trainCompleted = True
            depth+=1
        self.modelWeights = copy.deepcopy(wb_new[0])
        self.modelBias = copy.deepcopy(wb_new[1])
        print('train end>>>')
       
        return trainCompleted

    def calGrossClassifyPassRate(Y_p,Y):
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
    
    def testCalCostGrad(self,initTheta,X):
        return self.__calCostGradWithParams(initTheta,X)

    """
    initTheta = [[1.0],[2.0],[3,0]]
    """
    def __calCostGradWithParams(self,initTheta,X):
        [weights,bias] = GF.unpackWB(initTheta)
        [y_p,cacheA,cacheZ] = self.predictWithParams(X,weights,bias)
        cost = self.__calCost(y_p)
        self.__backwardNeurals(cacheA,cacheZ)
        grads = GF.packWB(self.derivLW,self.derivLB)
        return [cost,grads]
    
    def train3(self,monitorOption=None,outMonitorData=None):
        print('train begin>>>')
        trainCompleted = False
        if(self.trainX is None or self.trainY is None):
            print("invalid context, there is no train data setted")
            trainCompleted = False
            return trainCompleted
        self.__initTrain()
        options = tp.TrainOption()
        options.MaxIteration = self.maxIterationNum
        options.monitorOption.enable = True
        initTheta = GF.packWB(self.modelWeights,self.modelBias)
        # [theta,costs,iterNum,rates,grads] = GF.fmincg(self.__calCostGradWithParams,initTheta,options,self.trainX)
        # self.__mointor(monitorOption,outMonitorData,costs,rates,grads)
        
        [theta,mData] = GF.fminWithPolar(self.__calCostGradWithParams,initTheta,options,self.trainX)
        self.__mointor(monitorOption,outMonitorData,mData.costs,mData.rates,mData.grads)
        
        print(f"final train cost={outMonitorData.costs}")
        print(f"final train rates={outMonitorData.rates}")
        # print(f"bw={theta}")

        [self.modelWeights,self.modelBias]=GF.unpackWB(theta)
        trainCompleted = True
        return trainCompleted

    def test(self):
        testCompleted = False
        if(self.testX is None or self.testY is None):
            print("data of testX and testY not set yet call self.setData")
            return testCompleted
        if(self.modelWeights is None or self.modelBias is None):
            print("model not initialized call self.train")
            return testCompleted
        outputY,cacheA,cacheZ= self.predict(self.testX)
        passRate = SimpleFCModel.calGrossClassifyPassRate(outputY,self.testY)
        return(passRate,outputY)
    
    def getTrainAccuracy(self,Y_p):
        return ME.getBinaryClassifyAccuracy(self.trainY,Y_p)

    



        

