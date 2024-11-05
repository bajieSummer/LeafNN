from LeafNN.utils.Log import Log
import numpy as np
import LeafNN.core.LeafModels.ModelData as MD

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