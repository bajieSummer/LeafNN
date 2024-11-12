import demoInit
from LeafNN.core.Bases.MathMatrix import MathMatrix as MM
#import numpy as np
import os
from LeafNN.utils.PathUtils import PathUtils 
from LeafNN.utils.Log import Log
from LeafNN.utils.DataUtils import DataUtils as DUtils
from LeafNN.core.LeafModels.BaseClassifyModel import BaseClassifyModel as BCM
from LeafNN.core.LeafModels.NeuralLeaf import NeuralLeaf
import LeafNN.core.LeafModels.TrainMonitor as TMot
from LeafNN.utils.ModelVisualizer import ModelVisualizer as MV
LeafBaseTestTag = "LeafBaseModel_Test"

def readData1():
    file_name = "ex2data1.txt"
    dataPath = os.path.join(PathUtils.getDemoDatasPath(),file_name)
    return DUtils.readDataXYFromFile(dataPath)

def testReadSimpleData():
    data = readData1()
    X = data.X
    Y = data.Y
    XY = MM.hstack([X,Y])
    Log.Debug(LeafBaseTestTag,f"X shape={X.shape}, Y shape={Y.shape}")
    Log.Debug(LeafBaseTestTag,f"\n x=>\n{XY}")
    MV.plotData(X,Y)

    

def testPredictXBeforeTrain():
    # test case1: with ex2data1.txt
    # expected dldw = [[2.566, 2.647]] expected dldb =[[0.043] expected cost=0.218]
    # weights = [None]*1  # bias = [None]*1
    # weights[0] = np.array([[0.2],[0.2]])  
    # bias[0] = np.array([[-24.0]])
    data = readData1()
    X = data.X
    Y = data.Y
    Log.Debug(LeafBaseTestTag,f"X shape={X.shape}, Y shape={Y.shape}")
    wb_mats =[MM.array([[0.0],[0.0],[0.0]])]# -24.0 0.2,0.2
    wb = NeuralLeaf(wb_mats)
    (xn,xm) = X.shape
    model1 = BCM([xm,1],wb)
    Y_p = model1.predict(X,wb)
    XYY_p = MM.hstack([X,Y,Y_p])
    Log.Debug(LeafBaseTestTag,f" predict with certain values \n,{XYY_p}")
    indsStart = 2
    indsEnd = 10
    MV.plotDataWithTestCase(X,Y,X[indsStart:indsEnd],Y[indsStart:indsEnd],Y_p[indsStart:indsEnd])

def testCalCostAndGrad():
    # test case1:
    # expected dldw = [[2.566, 2.647]] expected dldb =[[0.043] expected cost=0.218]
    # wb = (-24.0,0.2,0.2)

    # test case2: with ex2data1.txt
    # expected dldw =[[-12.0092, -11.2628]] # expected dldb = [[-0.1000]] expected cost = 0.693
    # wb=(0.0,0.0,0.0)
    data = readData1()
    X = data.X
    Y = data.Y
    wb_mats =[MM.array([[0.0],[0.0],[0.0]])]
    wb = NeuralLeaf(wb_mats)
    (xn,xm) = X.shape
    model1 = BCM([xm,1],wb)
    cost = model1.calCost(wb,data)
    Log.Debug(LeafBaseTestTag,f"cost={cost}")
    grads = model1.calGrads(wb,data)
    Log.Debug(LeafBaseTestTag,f"grads=\n{grads}")
    [cost_1,grads_1] =model1.calCostAndGrads(wb,data)
    Log.Debug(LeafBaseTestTag,f"cost={cost_1},grads=\n{grads_1}")

def testTrain():
    data = readData1()
    MV.plotData(data.X,data.Y,"All Datas")
    wb_mats =[MM.array([[0.0],[0.0],[0.0]])]
    wb = NeuralLeaf(wb_mats)
    (xn,xm) = data.X.shape
    model1 = BCM([xm,1],wb)
    model1.trainOption.MaxIteration = 300
    model1.trainOption.trainRatio = 1.0
    model1.trainOption.validationRatio = 0.0
    model1.trainOption.testRatio = 0.0
    model1.setData(data)
    monitorOpiton=TMot.MonitorOption()
    monitorOpiton.enable = True
    monitorData=TMot.MonitorData()
    model1.train(monitorOpiton,monitorData)
    newWb = model1.wb
    MV.plotCostWithWB(monitorData.iterationInds,monitorData.costs,wb,newWb)
    startInds = 2
    endInds = 10
    testX = data.X[startInds:endInds]
    testY = data.Y[startInds:endInds]
    Y_p=model1.predict(testX,model1.wb)
    MV.plot2DDecisionBoundaryWithTestCase(newWb,data.X,data.Y,testX,testY,Y_p)
    # more debug info
    Log.Debug(LeafBaseTestTag,f"afterTrain: newWb=\n {newWb}")
    finalTrainCost = model1.calCost(newWb,model1.trainData)
    finalValidCost = model1.calCost(newWb,model1.validateData)
    finalTestCost = model1.calCost(newWb,model1.testData)
    Log.Debug(LeafBaseTestTag,f"afterTrain: trainCost={finalTrainCost},validCost={finalValidCost},testCost={finalTestCost}")
    Log.Debug(LeafBaseTestTag,f"monitorData costs=\n{monitorData.costs} \nrates=\n{monitorData.rates},")


MM.set_printoptions(precision=20, suppress=True)

def testMultiLayerNN():
    data = readData1()
    MV.plotData(data.X,data.Y,"All Datas")
    wb_mats =[]
    initV = 0.0
    wb_mats.append(MM.array([[1.0,1.0],[1.0,1.0],[1.0,1.0]])*initV)
    #wb_mats.append(MM.array([[1.0,1.0],[1.0,1.0],[1.0,1.0]])*initV)
    wb_mats.append(MM.array([[1.0],[1.0],[1.0]])*initV)
    wb = NeuralLeaf(wb_mats)
    (xn,xm) = data.X.shape
    model1 = BCM([xm,2,1],wb)
    model1.trainOption.MaxIteration = 300
    model1.trainOption.trainRatio = 1.0
    model1.trainOption.validationRatio = 0.0
    model1.trainOption.testRatio = 0.0
    model1.setData(data)
    monitorOpiton=TMot.MonitorOption()
    monitorOpiton.enable = True
    monitorData=TMot.MonitorData()
    model1.train(monitorOpiton,monitorData)
    newWb = model1.wb
    MV.plotCostWithWB(monitorData.iterationInds,monitorData.costs,wb,newWb)
    startInds = 2
    endInds = 10
    testX = data.X[startInds:endInds]
    testY = data.Y[startInds:endInds]
    Y_p=model1.predict(testX,model1.wb)
    MV.plot2DDecisionBoundaryWithTestCase(newWb,data.X,data.Y,testX,testY,Y_p)
    # more debug info
    Log.Debug(LeafBaseTestTag,f"afterTrain: newWb=\n {newWb}")
    finalTrainCost = model1.calCost(newWb,model1.trainData)
    finalValidCost = model1.calCost(newWb,model1.validateData)
    finalTestCost = model1.calCost(newWb,model1.testData)
    Log.Debug(LeafBaseTestTag,f"afterTrain: trainCost={finalTrainCost},validCost={finalValidCost},testCost={finalTestCost}")
    Log.Debug(LeafBaseTestTag,f"monitorData costs=\n{monitorData.costs} \nrates=\n{monitorData.rates},")
    
def main():
    Log.Debug(LeafBaseTestTag,"test case run")

    #testReadSimpleData()
    #testPredictXBeforeTrain()
    #testCalCostAndGrad()
    testMultiLayerNN()

    #data visual flow-> 
    # before train: dataXY --> 
    # train: train Costs/Iteration.(how many iterations, early stop?) -->
    #  after train: train_dataXY_Y_p, decisionBoundary,(train:pass rate.)-->
    

main()