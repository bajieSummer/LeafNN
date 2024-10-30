from LeafNN.utils.Log import Log
import numpy as np
LeafTag = "Leaf"
class Leaf:
    def __init__(self,matrixArray):
        Log.Debug(LeafTag,"leaf init")
        #self.length = len(matrixArray)
        # should I deepCopy?
        self.__matrixs = matrixArray

    def __add__(self,other):
        Log.Debug(LeafTag,"add operation")
        matrixResult = []
        success = False
        if isinstance(other,Leaf):
            if(other.getLayerSize()!=Leaf.getLayerSize()):
                success = False
                Log.Error(LeafTag,"Leaf add should have the same layers") 
            else:
                i = 0
                for mat in self.__matrixs:
                    matrixResult.append(mat+other.getMatrix(i))
                    i+=1
        else:
            Log.Error(LeafTag,"Leaf unknown type for Leaf add")
        if success:
            return Leaf(matrixResult)
        else:
            return None
    """
    warnings: be careful update of T
    """ 
    def T(self):
        if(self.__T == None):
            self.__T = []
            for mat in self.__matrixs:
                self.T.append(mat)
        return self.__T
    
    def __sub__(self,other):
        Log.Debug(LeafTag,"sub operation")
        matrixResult = []
        success = False
        if isinstance(other,Leaf):
            if(other.getLayerSize()!=Leaf.getLayerSize()):
                success = False
                Log.Error(LeafTag,"Leaf add should have the same layers") 
            else:
                i = 0
                for mat in self.__matrixs:
                    matrixResult.append(mat-other.getMatrix(i))
                    i+=1
        else:
            Log.Error(LeafTag,"Leaf unknown type for Leaf add")
        if success:
            return Leaf(matrixResult)
        else:
            return None

    # def getMatrix(self,indexLayer):
    #     return self.__matrixs[indexLayer]
    def __getitem__(self, layerIndex):
        """Return the item at the given index."""
        return self.__matrixs[layerIndex]

    # def __setitem__(self, layerIndex, value):
    #     """Set the item at the given index."""
    #     self.__matrixs[layerIndex] = value
    
    def getLayerSize(self):
        return len(self.__matrixs)

    def __multi__(self,other):
        matrixResult = []
        success = False
        if isinstance(other, (int, float)): 
            for mat in self.__matrixs:
                matrixResult.append(mat*other)
            success = True
        if isinstance(other,Leaf):
            if(other.getLayerSize()!=Leaf.getLayerSize()):
                success = False
                Log.Error(LeafTag,"Leaf multiplication should have the same layers")
            i = 0
            for mat in self.__matrixs:
                matrixResult.append(np.dot(mat,other.getMatrix(i)))
                i+=1
        else:
            Log.Error(LeafTag,"unknown type for multiplication")
        if(success):
            return Leaf(matrixResult)
        else:
            return None
        
    
    # the other Leaf should have the same laysize and same dimensions
    def LDot(self,other:'Leaf'):
        layerSize = len(self)
        if(layerSize!=len(other)):
            Log.Error(LeafTag,f"Ldot require same layersize, selfSize:{len(self)}otherSize:{len(self)}")
            return None
        Log.Debug(LeafTag,"LDot")
        result = 0.0
        for l in range(layerSize):
            result =result + np.dot(np.transpose(self[l]),other[l])
        return result
        
    def __repr__(self):
        strs = []
        for mat in self.__matrixs:
            strs.append(str(mat))
        return " ".join(strs)
    
    def __str__(self):
        return repr(self)
    

      
        


        



                
                
                