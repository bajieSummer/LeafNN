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
from PIL import Image

HandWriteClassifyTestTag = "HandWriteClassify_Test"

def createImgMatsFromData(X,ShowWidth=1024,pic_width=None):
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
                T =  MM.transpose(X[i].reshape(pic_width,pic_height))
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


def readData1():
    file_name = "ex3data1.mat"
    dataPath = os.path.join(PathUtils.getDemoDatasPath(),file_name)
    data = DUtils.readDataXYFromFile(dataPath)
    X = data.X
    Y = data.Y
    Log.Debug(HandWriteClassifyTestTag,f"datafirstpic: ==\n{X[0,:]}")
    return MD.ClassifyData(X,Y)

def Case1showData():
    data=readData1()
    X = data.X
    Y = data.Y
    Log.Debug(HandWriteClassifyTestTag,f"X shape={X.shape}, Y shape={Y.shape}")
    displayImgsFromX(X[0:1200,:])
    Log.Debug(HandWriteClassifyTestTag,f"Y=\n{Y[0:1200,:]}")


def train2LayerXY(Xt,Yt,layerSizeList,wb):
    data = MD.ClassifyData(Xt,Yt)
    model1 = BCM(layerSizeList,wb)
    #wb =model1.wb
    model1.trainOption.MaxIteration = 100
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


def trainOneVsAll(X,Y,numLabels):
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
        [wbk,mdata] = train2LayerXY(X,Yk,layerSizeList,wb_init)
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

    for i in range(m):
        max = -1
        for j in range(numLabels):
            if  Y[i][j]>max: #(Y[i][j]>=0.5) and
                max = Y[i][j]
                YRes[i] = j
    return YRes

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
    data=readData1()
    data = processDataY(data)
    # start =0
    # end = 2
    # data.X = data.X[start:end,:]
    # data.Y = data.Y[start:end,:]
    numLabels = 10
    wbs = trainOneVsAll(data.X,data.Y,numLabels)
    YRes = predictOneVsAll(data.X,wbs,numLabels)
    [rate,NonPassInds]=getPassRate(data.Y,YRes)
    Log.Debug(HandWriteClassifyTestTag,f"passRate={rate}")
    displayImgsFromX(data.X[NonPassInds,:],15*20)
    MV.plotYYpre(NonPassInds,data.Y,YRes,f"notPass Y and its prediction not_pass={len(NonPassInds)},total={len(YRes)}")

     
MM.set_printoptions(20,True)
def main():
    Log.Debug(HandWriteClassifyTestTag,"begin>>>>>>>")
    Case1showData()
    case2TrainOneVsAll()

main()