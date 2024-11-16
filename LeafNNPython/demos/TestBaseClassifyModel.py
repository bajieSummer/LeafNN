import demoInit
from LeafNN.Bases.MathMatrix import MathMatrix as MM
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

def readData2():
    file_name = "ex2data2.txt"
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
    #MV.plot2DDecisionBoundaryWithTestCase(newWb,data.X,data.Y,testX,testY,Y_p)
    MV.plot2DDecisionBoundary(newWb,data.X,data.Y)
    MV.plotDataWithTestCase(data.X,data.Y,testX,testY,Y_p)
    # more debug info
    Log.Debug(LeafBaseTestTag,f"afterTrain: newWb=\n {newWb}")
    finalTrainCost = model1.calCost(newWb,model1.trainData)
    finalValidCost = model1.calCost(newWb,model1.validateData)
    finalTestCost = model1.calCost(newWb,model1.testData)
    Log.Debug(LeafBaseTestTag,f"afterTrain: trainCost={finalTrainCost},validCost={finalValidCost},testCost={finalTestCost}")
    Log.Debug(LeafBaseTestTag,f"monitorData costs=\n{monitorData.costs} \nrates=\n{monitorData.rates},")


MM.set_printoptions(precision=20, suppress=True, threshold=MM.inf())

def testMultiLayerNN():
    data = readData1()
    data.X = DUtils.preprocessData(data.X,True,2)
    MV.plotData(data.X,data.Y,"All Datas")
    (xn,xm) = data.X.shape
    wb_mats =[]
    initV = 1.0
    #wb_mats.append(MM.array([[1.0,1.0],[1.0,1.0],[1.0,1.0]])*initV)
    #wb_mats.append(MM.array([[1.0,1.0],[1.0,1.0],[1.0,1.0]])*initV)
    #wb_mats.append(MM.array([[1.0],[1.0],[1.0]])*initV)
    layerSizeList = [xm,2,1]
    for l in range(len(layerSizeList)-1):
        wb_mats.append(MM.ones([layerSizeList[l]+1,layerSizeList[l+1]])*initV)
    wb = NeuralLeaf(wb_mats)    
    model1 = BCM(layerSizeList,wb)
    #wb =model1.wb
    model1.trainOption.MaxIteration = 100
    model1.trainOption.trainRatio = 1.0
    model1.trainOption.validationRatio = 0.0
    model1.trainOption.testRatio = 0.0
    model1.trainOption.regularLamada = 0.0
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
    #MV.plot2DDecisionBoundaryWithTestCase(newWb,data.X,data.Y,testX,testY,Y_p)
    MV.plotDataWithTestCase(data.X,data.Y,testX,testY,Y_p)
    # more debug info
    Log.Debug(LeafBaseTestTag,f"afterTrain: newWb=\n {newWb}")
    finalTrainCost = model1.calCost(newWb,model1.trainData)
    finalValidCost = model1.calCost(newWb,model1.validateData)
    finalTestCost = model1.calCost(newWb,model1.testData)
    Log.Debug(LeafBaseTestTag,f"afterTrain: trainCost={finalTrainCost},validCost={finalValidCost},testCost={finalTestCost}")
    Log.Debug(LeafBaseTestTag,f"monitorData costs=\n{monitorData.costs} \nrates=\n{monitorData.rates},")


def testReg():
    data = readData2()
    X_o = data.X*1.0
    Hdeg = 6
    isNorm = False
    Log.Debug(LeafBaseTestTag,f"originXShape={data.X.shape}")
    data.X = DUtils.preprocessData(data.X,isNorm,Hdeg)
    Log.Debug(LeafBaseTestTag,f"after process XShape={data.X.shape}")
    MV.plotData(data.X,data.Y,"All Datas")
    (xn,xm) = data.X.shape
    
    wb_mats =[]
    initV = 0.0
    #wb_mats.append(MM.array([[1.0,1.0],[1.0,1.0],[1.0,1.0]])*initV)
    #wb_mats.append(MM.array([[1.0,1.0],[1.0,1.0],[1.0,1.0]])*initV)
    #wb_mats.append(MM.array([[1.0],[1.0],[1.0]])*initV)
    layerSizeList = [xm,1]
    for l in range(len(layerSizeList)-1):
        wb_mats.append(MM.ones([layerSizeList[l]+1,layerSizeList[l+1]])*initV)
    wb = NeuralLeaf(wb_mats)    
    model1 = BCM(layerSizeList,wb)
    #wb =model1.wb
    model1.trainOption.MaxIteration = 100
    model1.trainOption.trainRatio = 1.0
    model1.trainOption.validationRatio = 0.0
    model1.trainOption.testRatio = 0.0
    model1.trainOption.regularLamada = 1.0
    model1.setData(data)
    monitorOpiton=TMot.MonitorOption()
    monitorOpiton.enable = True
    monitorData=TMot.MonitorData()
    [cost,grads] = model1.calCostAndGrads(wb,data)
    #Log.Debug(LeafBaseTestTag,f"costs={cost},grads={grads[0][:,0:3]}")
    model1.train(monitorOpiton,monitorData)
    newWb = model1.wb
    MV.plotCostWithWB(monitorData.iterationInds,monitorData.costs,wb,newWb)
    startInds = 0
    endInds = xn
    testX = data.X[startInds:endInds]
    testY = data.Y[startInds:endInds]
    Y_p=model1.predict(testX,model1.wb)
    #MV.plot2DDecisionBoundaryWithTestCase(newWb,data.X,data.Y,testX,testY,Y_p)
    [meshX,meshY] = DUtils.generateMeshPoints(X_o,Hdeg,0,1,newWb,model1.predict,isNorm,True)
    MV.plot2DDecisionBoundary(newWb,data.X,data.Y,meshX,meshY,[0.3,0.5,0.7])
    MV.plotDataWithTestCase(data.X,data.Y,testX,testY,Y_p)

    #MM.set_printoptions()
    Log.Debug(LeafBaseTestTag,f"aftercosts={cost},grads={grads}")
# import numpy as np
# import matplotlib.pyplot as plt
# def testMeshPoints():
#     a1 = np.linspace(-1,1,20)
#     a2 = np.linspace(-10,10,20)
#     p1 = np.ones([20,2])*10
#     p1[:,0]=a1
#     p1[:,1]=a2
#     print(a1.shape)
#     print(a2.shape)
#     [X,Y] = np.meshgrid(a1,a2)
#     print(X)
#     print(Y)
#     Z = X*X + Y*Y -1
#     plt.figure()
#     plt.contour(p1[:,0], p1[:,1], Z, levels=[0], colors='blue', linewidths=2)
#     plt.show()
#     t1 = MM.ones([3,2])*2
#     tf = MM.ones([3,1])*3
#     t2 = t1*1.0
#     t2[0][0] = 10
#     t2[:,1] = tf.flatten()
#     print(t1)
#     print(t2)
    
#testMeshPoints()
def main():
    Log.Debug(LeafBaseTestTag,"test case run")

    #testReadSimpleData()
    #testPredictXBeforeTrain()
    #testCalCostAndGrad()
    #testTrain()
    #testMultiLayerNN()
    testReg() 

main()