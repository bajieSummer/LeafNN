'''
Author: Sophie
email: bajie615@126.com
Date: 2024-07-10 14:09:06
Description: file content
'''
from abc import abstractmethod
from LeafNN.core.DLModels.TrainOptions import GradientOptions
from LeafNN.utils.PathUtils import PathUtils
from LeafNN.utils.Log import Log
from LeafNN.utils.Log import LogOption
from LeafNN.utils.Log import LogLevel
from LeafNN.utils.Log import LogTag
class BaseModel:
    """
    this is a basic abstract class for BaseModel
    """
    def __init__(self,layerSize,layerNodeSizeList):
        """
        layerSize : how many layers of the model
        layerNodeSizeList: how many nodes of each layer eg: [2,3,5] which means there are 
        2 nodes for input layer 3 nodes in middle layer, 5 nodes in output layer
        """
        print("base")
        if not isinstance(layerNodeSizeList,list):
            raise TypeError("layerNodeSizeList must be a list")
        if len(layerNodeSizeList) != layerSize:
            raise ValueError("layerSize must be the same size as length of layerNodeSizeList")
        self.layerSize = layerSize
        self.layerNodeSizeList = layerNodeSizeList
        
        self.trainInput = None
        self.modelWeights = None
        self.modelBias = None

        self.activeFunc = None
        self.derivActiveFunc = None
        self.lossFunc = None
        self.derivLossFunc = None

        self.gradientMethod = None
        # ? we should call this at initialization of the whole modules
        Log.config(PathUtils.getLogFolderPath(),LogOption())

        
   
    def setActiveFunc(self,activeFunc):
        self.activeFunc = activeFunc

    def setLossFunc(self,lossFunc):
        self.lossFunc = lossFunc

    def setGradientMethod(self,gradientMethod):
        self.gradientMethod = gradientMethod
    
    def printModelInfo(self):
        msg=f"Model get {self.layerSize} layers, each layer has {self.layerNodeSizeList} nodes"
        Log.Critical(LogTag.DLModels,msg)

    @abstractmethod
    def train(self):
        pass

