import demoInit
import os
from LeafNN.utils.PathUtils import PathUtils 
import LeafNN.core.DLModels.TrainOptions as tp
import LeafNN.core.DLModels
from LeafNN.core.DLModels.BaseModel import BaseModel
from LeafNN.core.DLModels.SimpleFCModel import SimpleFCModel
import numpy as np
import matplotlib.pyplot as plt
import copy


def readTrainDataXYFromFile(filePath):
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
    return(dataX,dataY)

def preProcessData(dataX):
    dataX =dataX
    return dataX

def plotData(X,Y):
    # Separate the data based on the values of y
    x_y0 = X[Y.flatten() == 0]
    x_y1 = X[Y.flatten() == 1]

    # Plot the points with different shapes or colors based on the value of y
    plt.scatter(x_y0[:, 0], x_y0[:, 1], color='blue', marker='s', label='y=0')  # Squares for y=0
    plt.scatter(x_y1[:, 0], x_y1[:, 1], color='blue', marker='^', label='y=1')   # Triangles for y=1

    # Add labels and title
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('x has 2 features. classification plotting')

    # Add a legend
    plt.legend()
    # Display the plot
    plt.show()


def plotDataWithTestCase(X,Y,xt,yt_p,yt):
    # Separate the data based on the values of y
    x_y0 = X[Y.flatten() == 0]
    x_y1 = X[Y.flatten() == 1]

    # Plot the points with different shapes or colors based on the value of y
    plt.scatter(x_y0[:, 0], x_y0[:, 1], color='blue', marker='s', label='y=0')  # Squares for y=0
    plt.scatter(x_y1[:, 0], x_y1[:, 1], color='blue', marker='^', label='y=1')   # Triangles for y=1

    markert = 's'
    if yt_p > 0.5 : markert ='^'
    colort = 'green'
    print(f"yt_p={yt_p},yt={yt}")
    if(np.abs(yt_p - yt)>0.5): colort='red'
    plt.scatter(xt[:, 0], xt[:, 1], color=colort, marker=markert)  
    
    
    # Add labels and title
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('x has 2 features. classification plotting')

    # Add a legend
    plt.legend()
    # Display the plot
    plt.show()

def plotCost(inds,costs):
    plotCostWithWB(inds,costs)

def plotCostWithWB(inds,costs,wb=None):
    length = len(costs)
    if length <= 0:
        print("error:costs is empty")
        return

    lastCost = costs[length-1]
    title = f"cost/iterations lastCost={lastCost}"
    if(wb is not None):
        title=f"{title},initW={wb[0][0]},initB={wb[1][0]}"
    plt.title(title)
    plt.scatter(inds,costs,color='orange',marker='s',label ='cost')
    # Add a legend
    plt.legend()
    # Display the plot
    plt.show()

def plotDecisionBoundary(weights,bias,X,Y):
    # boundary h(z(x,cita)) > 0.5 ->1.0 h(z(x,cita))<0.5 -->0.0
    # z(x,cita) >0 -->1.0 z(x,cita)<0 -->0.0 z(x,cita) = 0 -->boundary
    # first get dataX(x0->max,->min) z(x1,x2) = 0 ->x2min,x2max
    # Separate the data based on the values of y
    x_y0 = X[Y.flatten() == 0]
    x_y1 = X[Y.flatten() == 1]

    # Plot the points with different shapes or colors based on the value of y
    plt.scatter(x_y0[:, 0], x_y0[:, 1], color='blue', marker='s', label='y=0')  # Squares for y=0
    plt.scatter(x_y1[:, 0], x_y1[:, 1], color='blue', marker='^', label='y=1')   # Triangles for y=1
    x0_max = X[0][0]
    x0_min = X[0][0]
    for xi in X:
        if(xi[0]>x0_max):
            x0_max = xi[0]
        if(xi[0]<x0_min):
            x0_min = xi[0]
    x1_max = (x0_max*weights[0][0][0] + +bias[0][0][0])/(-1.0*(weights[0][1][0]+np.finfo(float).eps))
    x1_min = (x0_min*weights[0][0][0] + +bias[0][0][0])/(-1.0*(weights[0][1][0]+np.finfo(float).eps))
    x0min_max =[x0_min,x0_max]
    x1min_max =[x1_min,x1_max] 
    plt.plot(x0min_max,x1min_max,linestyle="-")
    
    # Add labels and title
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('x has 2 features. classification plotting')

    # Add a legend
    plt.legend()
    # Display the plot
    plt.show()

def plotLearnRates(monitorData:tp.MonitorData):
    if(monitorData.rates):
        plt.title("learning rates")
        plt.scatter(monitorData.iterationInds[0:len(monitorData.rates)],monitorData.rates,color='orange',marker='s',label ='learnRate')
    # Add a legend
    plt.legend()
    # Display the plot
    plt.show()

    

def initTestBiasAndWeights():
    
    # test case1: with ex2data1.txt
    # expected dldw = [[2.566, 2.647]] expected dldb =[[0.043] expected cost=0.218]
    # weights = [None]*1
    # weights[0] = np.array([[0.2],[0.2]])  
    # bias = [None]*1
    # bias[0] = np.array([[-24.0]])
    
    # test case2: with ex2data1.txt
    # expected dldw =[[-12.0092, -11.2628]] # expected dldb = [[-0.1000]] expected cost = 0.693
    # weights = [None]*1
    # weights[0] = np.zeros([2,1])  
    # bias = [None]*1
    # bias[0] = np.zeros([1,1])
    # expect b = -25.161\n w1 =0.206\n w2 = 0.201 with 400 interation

    weights = [None]*1
    weights[0] = np.array([[100.0],[100.0]]) #[-0.78685205],[-0.96436821] # 0.2,0.2  # 10.0,15.0 strange problem
    bias = [None]*1 
    bias[0] = np.array([[1000.0]])  #-10.0 #[100.0]
     
    # testcase3: wrong dldw and checkdw w =1.20092166,1.12628422,b=1.12628422
    # weights =[np.array([[1.20092166],
    #    [1.12628422]])]
    # bias = [np.array([[0.01]])]
    return (weights,bias)
np.set_printoptions(precision=20, suppress=True)
# readfiles:
file_name = "ex2data1.txt"
dataPath = os.path.join(PathUtils.getDemoDatasPath(),file_name)
(dataX,dataY) = readTrainDataXYFromFile(dataPath)
dataX = preProcessData(dataX)

(xn,xm) = dataX.shape
print("dataX ,dataY")
print(np.hstack([dataX,dataY]))
logisticModel = SimpleFCModel(2,[xm,1])
logisticModel.printModelInfo()
logisticModel.trainProportion = 1.0
logisticModel.testProportion = 0.0
logisticModel.learnRate =  0.0002 # 0.0015625#
logisticModel.setData(dataX,dataY)
logisticModel.initWeights = initTestBiasAndWeights
logisticModel.maxIterationNum = 400
logisticModel.enableGradientCheck = False
logisticModel.gradientCheckFrequency = 1
logisticModel.enableEarlyStop = False
monitorOpiton=tp.MonitorOption()
monitorOpiton.enable = True
monitorData=tp.MonitorData()
[initWeights,initBias] = initTestBiasAndWeights()
complete = logisticModel.train3(monitorOpiton,monitorData)
[cost,grads]= logisticModel.testCalCostGrad(np.array([[98.05141701],
 [-0.78685205],
 [-0.96436821]]),logisticModel.trainX)
print("costs_test_example=>",cost)
[y_out,ca,cz] = logisticModel.predict(logisticModel.trainX)
print(f"train accuracy={logisticModel.getTrainAccuracy(y_out)}")
print("train complete?:",complete)
print("weights=>")
print(logisticModel.modelWeights)
print("bias=>")
print(logisticModel.modelBias)

# 35.62,46.8 0
# 42.07545454 78.844786,   0.54270337  0. 
xt = np.array([[35.62,46.8]])
yt = 0
(yt_p,ca,cz) = logisticModel.predict(xt)
plotDataWithTestCase(dataX,dataY,xt,yt_p,yt)

plotCostWithWB(monitorData.iterationInds,monitorData.costs,[initWeights,initBias])
plotDecisionBoundary(logisticModel.modelWeights,logisticModel.modelBias,logisticModel.trainX,logisticModel.trainY)
#plotLearnRates(monitorData)
# print("monitor data")
# print(monitorData.iterationInds)
# print(monitorData.costs)
# (rate,testY_p) = logisticModel.test()
# print("passRate=",rate)
# print(f"testY_p shape={testY_p.shape},testY shape={logisticModel.testY.shape}")
# print(np.hstack([logisticModel.testX,testY_p,logisticModel.testY]))

# testInd = 0
# z = np.matmul(logisticModel.testX[testInd],logisticModel.modelWeights)+logisticModel.modelBias
# print(f"z={z},y_p={SimpleFCModel.defaultActive(z)}")
# (zp,cachea,cachez) = logisticModel.predict(logisticModel.testX[testInd:testInd+1,:])
# print(zp)






