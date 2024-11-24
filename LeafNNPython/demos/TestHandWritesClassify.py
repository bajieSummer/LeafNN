# test pics classify handwrite classification
import demoInit
import os
from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.utils.PathUtils import PathUtils 
from LeafNN.utils.Log import Log
from LeafNN.utils.DataUtils import DataUtils as DUtils
from LeafNN.core.LeafModels.BaseClassifyModel import BaseClassifyModel as BCM
from LeafNN.core.LeafModels.NeuralLeaf import NeuralLeaf
import LeafNN.core.LeafModels.TrainMonitor as TMot
from LeafNN.utils.ModelVisualizer import ModelVisualizer as MV
import LeafNN.core.LeafModels.ModelData as MD
from LeafNN.utils.ImageUtils import ImageUtils


HandWriteClassifyTestTag = "HandWriteClassify_Test"

def readMatData1(file_name,transpose=False,picW=None):
    #file_name = "ex3data1.mat"
    dataPath = os.path.join(PathUtils.getDemoDatasPath(),file_name)
    data = DUtils.readDataXYFromFile(dataPath,transpose,picW)
    return data

def readWB(file_name):
    dataPath = os.path.join(PathUtils.getDemoDatasPath(),file_name)
    return DUtils.readWB(dataPath)

def saveWB1(leaf,file_name):
    dataPath = os.path.join(PathUtils.getDemoDatasPath(),file_name)
    data = DUtils.writeWB(leaf,dataPath)

def Case1showData():
    data=readMatData1("ex3data1.mat",True)
    X = data.X
    Y = data.Y
    Log.Debug(HandWriteClassifyTestTag,f"X shape={X.shape}, Y shape={Y.shape}")
    ImageUtils.displayImgsFromX(X[0:1200,:],1024,2)
    Log.Debug(HandWriteClassifyTestTag,f"Y=\n{Y[0:1200,:]}")


def train2LayerXY(Xt,Yt,layerSizeList,wb,maxIteration):
    data = MD.ClassifyData(Xt,Yt)
    model1 = BCM(layerSizeList,wb)
    #wb =model1.wb
    model1.trainOption.MaxIteration = maxIteration
    model1.trainOption.trainRatio = 1.0
    model1.trainOption.validationRatio = 0.0
    model1.trainOption.testRatio = 0.0
    model1.trainOption.regularLamada = 0.1
    model1.setData(data)
    monitorOpiton=TMot.MonitorOption()
    monitorOpiton.enable = True
    monitorData=TMot.MonitorData()
    model1.train(monitorOpiton,monitorData)
    return [model1.wb,monitorData]


def trainOneVsAll(X,Y,numLabels,maxIteration):
    [m,n] = X.shape
    [m,n2] = Y.shape
    wbs=[]
    wb_mats_init =[]
    initV = 0.0
    layerSizeList = [n,1]
    for l in range(len(layerSizeList)-1):
        wb_mats_init.append(MM.ones([layerSizeList[l]+1,layerSizeList[l+1]])*initV)
    wb_init = NeuralLeaf(wb_mats_init)  
    for k in range(numLabels):
        Yk = MM.zeros(Y.shape)
        Yk[Y==k] = 1.0
        [wbk,mdata] = train2LayerXY(X,Yk,layerSizeList,wb_init,maxIteration)
        wbs.append(wbk)
        Log.Debug(HandWriteClassifyTestTag,f"k={k} wb=\n{wbs[k]}")
        MV.plotCostWithWB(mdata.iterationInds,mdata.costs,wb_init,wbk)
    return wbs

def predictOneVsAll(X,wbs,numLabels):
    [m,n] = X.shape
    layerSizeList = [n,1]
    Y = None
    for i in range(numLabels):
        model1 = BCM(layerSizeList,wbs[i])
        Yi = model1.predict(X,wbs[i])
        if Y is None:
            Y = Yi
        else:
            Y = MM.hstack([Y,Yi])
    YRes = MM.ones([m,1])*(-1)
    Y_prob = MM.ones([m,1])*0.0
    for i in range(m):
        max = -1
        for j in range(numLabels):
            if  Y[i][j]>max: #(Y[i][j]>=0.5) and
                max = Y[i][j]
                YRes[i] = j
                Y_prob[i] = max
    return [YRes,Y_prob]

def getPassRate(Y,YRes):
    [m,n] = Y.shape
    numPass = 0
    NonpassInds = []
    for i in range(m):
        if Y[i] == YRes[i]:
            numPass +=1
        else:
            NonpassInds.append(i)
    return [numPass/m,NonpassInds]

def processDataY(data):
    resData=MD.ClassifyData(data.X,data.Y)
    (m,n) = resData.Y.shape
    for i in range(m):
        for j in range(n):
            if (resData.Y[i][j]==10):
                resData.Y[i][j] = 0
    return resData
        

def case2TrainOneVsAll():
    data=readMatData1("ex3data1.mat",True)
    data = processDataY(data)
    # start =0
    # end = 2
    # data.X = data.X[start:end,:]
    # data.Y = data.Y[start:end,:]
    numLabels = 10
    maxIteraion = 100
    wbs = trainOneVsAll(data.X,data.Y,numLabels,maxIteraion)
    i = 0
    for wb in wbs:
        saveWB1(wb,f"wb{i}_ex3data1.json")
        i+=1
    [YRes,Yprob] = predictOneVsAll(data.X,wbs,numLabels)
    [rate,NonPassInds]=getPassRate(data.Y,YRes)
    Log.Debug(HandWriteClassifyTestTag,f"passRate={rate}")
    ImageUtils.displayImgsFromX(data.X[NonPassInds,:],15*20,2)
    MV.plotYYpre(NonPassInds,data.Y,YRes,f"notPass Y and its prediction not_pass={len(NonPassInds)},total={len(YRes)}")

def case3TestHandWritingsWithTrainedWB():
    numLabels = 10
    wbs=[]
    trainData = readMatData1("ex3data1.mat",True)
    tX = trainData.X
    tY = trainData.Y
    tY[tY==10] = 0
    for i in range(numLabels):
        #saveWB1(wb,f"wb{i}_ex3data1.json")
        wb = readWB(f"wb{i}_ex3data1.json")
        wbs.append(wb)
    #start = 1100
    #end = 1200
    #xindices = MM.arange(end-start) + start
    xindices = MM.randIndices(len(tX),100)
    [YRes,Yprob] = predictOneVsAll(tX[xindices,:],wbs,numLabels)
    Y_indices = MM.arange(YRes.shape[0])
    ImageUtils.displayImgsFromX(tX[xindices,:],400,2)
    MV.plotYYpre(Y_indices,tY[xindices,:],YRes,f"predict")


      
def case4TestPredictNumbersFromPics():
    #createTestXYFromPics(folderPath)
    folder_name = "testPics1"
    testPicsFolderPath = os.path.join(PathUtils.getDemoDatasPath(),folder_name)
    testXY = ImageUtils.createXYDataFromNumberPics(testPicsFolderPath,False,True,20,20)
    numLabels = 10
    wbs=[]
    for i in range(numLabels):
        #saveWB1(wb,f"wb{i}_ex3data1.json")
        wb = readWB(f"wb{i}_ex3data1.json")
        wbs.append(wb)
    [YRes,Yprob] = predictOneVsAll(testXY.X,wbs,numLabels)
    size_y = len(YRes)
    xindices = MM.arange(size_y)
    ImageUtils.displayImgsFromX(testXY.X,400,2)
    MV.plotYYpre(xindices,testXY.Y,YRes,f"numberPicsPredict,size={size_y}")
    Log.Debug(HandWriteClassifyTestTag,f"y-y_pre-probilitys\n{MM.hstack([testXY.Y,YRes,Yprob])}")

# def transposeX(X,picW,picH):
#     [m,n] = X.shape
#     Xres = X
#     for i in range(m):
#         xi = X[i,:].reshape([picW,picH])
#         xi = MM.transpose(xi)
#         Xres[i,:] = xi.flatten()
#     return Xres

def case5TrainAll2All():
    origData = readMatData1("ex3data1.mat",True)
    # processY
    X = origData.X
    oY = origData.Y
    [m,n] = oY.shape
    oY[oY==10] = 0
    numLabels = 10
    Y = MM.zeros([m,numLabels])
    for i in range(m):
        yi = int(oY[i,0].flatten())
        Y[i,yi] = 1
    # inital wb
    [m,n] = X.shape
    [m,n2] = Y.shape
    wb_mats_init =[]
    initV = 0.0
    layerSizeList = [n,25,n2]
    for l in range(len(layerSizeList)-1):
        wb_mats_init.append(MM.ones([layerSizeList[l]+1,layerSizeList[l+1]])*initV)
    wb_init = NeuralLeaf(wb_mats_init)
    # createModels
    model1 = BCM(layerSizeList)
    model1.trainOption.MaxIteration = 100
    model1.trainOption.trainRatio = 1.0
    model1.trainOption.validationRatio = 0.0
    model1.trainOption.testRatio = 0.0
    model1.trainOption.regularLamada = 10.0
    model1.setData(MD.ClassifyData(X,Y))
    monitorOpiton=TMot.MonitorOption()
    monitorOpiton.enable = True
    monitorData=TMot.MonitorData()
    model1.train(monitorOpiton,monitorData)
    MV.plotCostWithWB(monitorData.iterationInds,monitorData.costs,wb_init,model1.wb)
    saveWB1(model1.wb,"wbs_ex3data1.json")
    MV.plotGradientsSquare(monitorData.iterationInds,monitorData.grads,"Gradients Strength Trend in Training")

    YRes = model1.predict(X,model1.wb)
    YRes_process = MM.argmax(YRes,1)
    passedNums =0
    for i in range(m):
        if(YRes_process[i]==oY[i]):
            passedNums+=1
    passRate = passedNums/m
    Log.Debug(HandWriteClassifyTestTag,f"samples={m},passRate={passRate}")
    #  

def case6predictTrainDataWithWB_all2all():
    origData = readMatData1("ex3data1.mat",True)
    wb = readWB("wbs_ex3data1.json")
    outSide = False
    # origData = readMatData1("ex3data1.mat",False)
    # wb = readWB("ex3weights.mat")
    # outSide = True
    indStart =4600
    indEnd = 4700
    X = origData.X[indStart:indEnd,:]
    Y = origData.Y[indStart:indEnd,:]
    Y[Y==10] = 0
    #Log.Debug(HandWriteClassifyTestTag,f"Y=\n{Y}")
    numLabels = 10
    [m,n] = X.shape
    [m,n2] = Y.shape
    numLabels = 10
    
    layerSizeList=[n,25,numLabels]
    model1 = BCM(layerSizeList,wb)
    YRes = model1.predict(X,wb)
    Log.Debug(HandWriteClassifyTestTag,f"YRes=\n{YRes}")
    YRes_process = MM.argmax(YRes,1).reshape(-1,1)
    if(outSide):
        YRes_process = YRes_process+1
        YRes_process[YRes_process == 10] = 0
    YCompare = MM.hstack([Y,YRes_process])*1.0
    Log.Debug(HandWriteClassifyTestTag,f"Y-YRes_process=\n{YCompare}")
    passedNums =0
    for i in range(m):
        if(YRes_process[i]==Y[i]):
            passedNums+=1
    ImageUtils.displayImgsFromX(X,400,2)
    folderPath = os.path.join(PathUtils.getDemoDatasPath(),"savePics")
    ImageUtils.saveNumberImgsFromXYData(folderPath,X,Y,20,20)
    passRate = passedNums/m
    Log.Debug(HandWriteClassifyTestTag,f"samples={m},passRate={passRate}")


def case7predictMyHandWritesWithWB_all2all():
    folder_name = "testPics1"
    # wb = readWB("ex3weights.mat")
    # outSide = True
    wb = readWB("wbs_ex3data1.json")
    outSide =False
    
    testdataFolderPath = os.path.join(PathUtils.getDemoDatasPath(),folder_name)
    originTestXY = ImageUtils.createXYDataFromNumberPics(testdataFolderPath,outSide,False,20,20)
    X = originTestXY.X
    oY = originTestXY.Y
    numLabels = 10
    [m,n] = X.shape
    [m,n2] = oY.shape
    #oY[oY==10] = 0
    Y = MM.zeros([m,numLabels])
    for i in range(m):
        yi = int(oY[i,0].flatten())
        Y[i,yi] = 1
    # inital wb
   
    layerSizeList=[n,25,numLabels]
    model1 = BCM(layerSizeList,wb)
    YRes = model1.predict(X,wb)
    YRes_process = MM.argmax(YRes,1).reshape(m,1)
    if(outSide):
        YRes_process = YRes_process+1
        YRes_process[YRes_process == 10] = 0
    passedNums =0
    for i in range(m):
        if(YRes_process[i]==originTestXY.Y[i]):
            passedNums+=1
    passRate = passedNums/m
    Log.Debug(HandWriteClassifyTestTag,f"samples={m},passRate={passRate}")
    ImageUtils.displayImgsFromX(X,400,2)
    indices = MM.arange(m)
    MV.plotYYpre(indices,originTestXY.Y,YRes_process,"img-numbers identify")
    Log.Debug(HandWriteClassifyTestTag,f"YResProcess-Y{MM.hstack([YRes_process,oY])}")

     
MM.set_printoptions(20,True)
def main():
    Log.Debug(HandWriteClassifyTestTag,"begin>>>>>>>")
    #Case1showData()
    #case2TrainOneVsAll()
    #case3TestHandWritingsWithTrainedWB()
    #case4TestPredictNumbersFromPics()
    #case5TrainAll2All()
    #case6predictTrainDataWithWB_all2all()
    case7predictMyHandWritesWithWB_all2all()

main()