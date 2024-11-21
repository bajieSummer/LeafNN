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
import re
from PIL import Image
from pathlib import Path

HandWriteClassifyTestTag = "HandWriteClassify_Test"

def createImgMatsFromData(X,ShowWidth=1024,isTranspose=False,pic_width=None):
    [m,n] = X.shape
    if pic_width is None:
        pic_width = int(MM.sqrt(n))
    pic_height = int(n/pic_width)
    Log.Debug(HandWriteClassifyTestTag,f"X shape is{m} {n}, Width={pic_width},height={pic_height}")
    imgXs =[]
    #imgXs.append(X[0].reshape(pic_width,pic_height))
    columns = int(ShowWidth/pic_width)
    blanks = MM.ones([pic_width,pic_height])
    i = 0
    while i < m:
        imgXi = None
        j = 0
        while j < columns:
            if i >=m:
                imgXi = MM.hstack([imgXi,blanks])
            else:
                T = X[i].reshape(pic_width,pic_height)
                if isTranspose:
                    T =  MM.transpose(T)
                if imgXi is None:
                    imgXi = T
                else:
                    imgXi = MM.hstack([imgXi,T])
            j+=1
            i+=1
        imgXs.append(imgXi)
    imgsMat = None
    for img in imgXs:
        if imgsMat is None:
            imgsMat = img
        else:
            imgsMat=MM.vstack([imgsMat,img])
    return imgsMat
    
    
def displayImgsFromX(X,ShowWidth=1024,imgWidth=None):
    imgMats = createImgMatsFromData(X,ShowWidth,imgWidth)
    img = Image.fromarray(imgMats*255)
    osize = img.size
    scaled_image = img.resize((osize[0]*2,osize[1]*2), Image.LANCZOS) 
    scaled_image.show()

def transposeMatX(X):
    [m,n] = X.shape
    picW = int(MM.sqrt(n))
    resX = MM.ones(X.shape)
    for i in range(m):
        rxi = X[i,:].reshape([picW,int(n/picW)])
        resX[i,:] = MM.transpose(rxi).flatten()
    return resX



def readMatData1(file_name,transpose=False):
    #file_name = "ex3data1.mat"
    dataPath = os.path.join(PathUtils.getDemoDatasPath(),file_name)
    data = DUtils.readDataXYFromFile(dataPath)
    X = data.X
    X = transposeMatX(X)
    Y = data.Y
    Log.Debug(HandWriteClassifyTestTag,f"datafirstpic: ==\n{X[0,:]}")
    return MD.ClassifyData(X,Y)

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
    displayImgsFromX(X[0:1200,:])
    Log.Debug(HandWriteClassifyTestTag,f"Y=\n{Y[0:1200,:]}")


def train2LayerXY(Xt,Yt,layerSizeList,wb,maxIteration):
    data = MD.ClassifyData(Xt,Yt)
    model1 = BCM(layerSizeList,wb)
    #wb =model1.wb
    model1.trainOption.MaxIteration = maxIteration
    model1.trainOption.trainRatio = 1.0
    model1.trainOption.validationRatio = 0.0
    model1.trainOption.testRatio = 0.0
    model1.trainOption.regularLamada = 1.0
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
    displayImgsFromX(data.X[NonPassInds,:],15*20)
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
    displayImgsFromX(tX[xindices,:],400)
    MV.plotYYpre(Y_indices,tY[xindices,:],YRes,f"predict")

def createTestXYFromBaseNumberPics(folderPath,isResize=False,picW=None,picH=None):
    directory = Path(folderPath)
    file_paths = list(directory.rglob('*'))
    testX = None
    testY = None
    for fpath in file_paths:
        image = Image.open(fpath)
        if image is None:
            continue
        if(isResize and picW and picH):
            image = image.resize((picW,picH))
        image = image.convert('L')
        filename = os.path.basename(fpath)
        numbers = re.findall(r'\d+', filename)
        number = None
        if numbers:
            number = float(numbers[0])
        else:
            continue
        image_array = MM.array(image)
        #image_array = MM.transpose(MM.array(image))
        Log.Debug(HandWriteClassifyTestTag,f"img=\n{image_array}")
        image_array = image_array.flatten().reshape(1, -1)
        # to do why
      
        if(testX is None):
            testX = image_array
        else:
            testX = MM.vstack([testX,image_array])
        curY = MM.ones([1,1])*number
        if(testY is None):
            testY = curY
        else:
            testY = MM.vstack([testY,curY])
    return MD.ClassifyData(testX/255.0,testY)
      
def case4TestPredictNumbersFromPics():
    #createTestXYFromPics(folderPath)
    folder_name = "testPics1"
    testdataFolderPath = os.path.join(PathUtils.getDemoDatasPath(),folder_name)
    testXY = createTestXYFromBaseNumberPics(testdataFolderPath,True,20,20)
    numLabels = 10
    wbs=[]
    for i in range(numLabels):
        #saveWB1(wb,f"wb{i}_ex3data1.json")
        wb = readWB(f"wb{i}_ex3data1.json")
        wbs.append(wb)
    [YRes,Yprob] = predictOneVsAll(testXY.X,wbs,numLabels)
    size_y = len(YRes)
    xindices = MM.arange(size_y)
    displayImgsFromX(testXY.X,400)
    MV.plotYYpre(xindices,testXY.Y,YRes,f"numberPicsPredict,size={size_y}")
    Log.Debug(HandWriteClassifyTestTag,f"y-y_pre-probilitys\n{MM.hstack([testXY.Y,YRes,Yprob])}")

        
     
MM.set_printoptions(20,True)
def main():
    Log.Debug(HandWriteClassifyTestTag,"begin>>>>>>>")
    #Case1showData()
    #case2TrainOneVsAll()
    #case3TestHandWritingsWithTrainedWB()
    case4TestPredictNumbersFromPics()

main()