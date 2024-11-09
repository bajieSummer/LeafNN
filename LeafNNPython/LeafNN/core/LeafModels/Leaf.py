from LeafNN.core.Bases.MathMatrix import MathMatrix as MM
from LeafNN.utils.Log import Log
#import numpy as np
LeafTag = "Leaf"
class Leaf:
    def __init__(self,matrixArray):
        #Log.Debug(LeafTag,"leaf init")
        #self.length = len(matrixArray)
        # should I deepCopy?
        self._matrixs = matrixArray

    def __add__(self,other):
        #Log.Debug(LeafTag,"add operation")
        matrixResult = []
        success = True
        if isinstance(other,Leaf):
            if(self.getLayerSize()!=other.getLayerSize()):
                success = False
                Log.Error(LeafTag,"Leaf add should have the same layers") 
            else:
                i = 0
                for mat in self._matrixs:
                    matrixResult.append(mat+other[i])
                    i+=1
        else:
            Log.Error(LeafTag,"Leaf unknown type for Leaf add")
        if success:
            return self._createInstance(matrixResult)
        else:
            return None
    """
    warnings: be careful update of T todo
    """ 
    def T(self):
        if(self.__T == None):
            T = []
            for mat in self._matrixs:
                T.append(MM.transpose(mat))
            self.__T = self._createInstance(T)
        return self.__T
    
    def __sub__(self,other):
        #Log.Debug(LeafTag,"sub operation: a-other")
        matrixResult = []
        success = True
        if isinstance(other,Leaf):
            if(other.getLayerSize()!=Leaf.getLayerSize()):
                success = False
                Log.Error(LeafTag,"Leaf add should have the same layers") 
            else:
                i = 0
                for mat in self._matrixs:
                    matrixResult.append(mat-other.getMatrix(i))
                    i+=1
        else:
            Log.Error(LeafTag,"Leaf unknown type for Leaf sub")
        if success:
            return self._createInstance(matrixResult)
        else:
            return None

    def __neg__(self):
        matrixResult = []
        for mat in self._matrixs:
            matrixResult.append(-1.0*mat)
        return self._createInstance(matrixResult)
    
    def __radd__(self,other):
        #Log.Debug(LeafTag,"add operation other+self")
        return other + self
    
    def __rsub__(self,other):
        #Log.Debug(LeafTag,"sub operation: other-self")
        matrixResult = []
        success = True
        if isinstance(other,Leaf):
            if(other.getLayerSize()!=Leaf.getLayerSize()):
                success = False
                Log.Error(LeafTag,"Leaf add should have the same layers") 
            else:
                i = 0
                for mat in self._matrixs:
                    matrixResult.append(other.getMatrix(i)-mat)
                    i+=1
        else:
            Log.Error(LeafTag,"Leaf unknown type for Leaf rsub")
        if success:
            return self._createInstance(matrixResult)
        else:
            return None
    
    def __multiplyScalar(self,other):
        matrixResult = []
        for mat in self._matrixs:
            matrixResult.append(mat*other)
        return self._createInstance(matrixResult)

    def __mul__(self,other):
        result = 0
        success = True
        # if other like this? Leaf([np.array([[1.0]]]))
        if MM.isNum(other): 
            return self.__multiplyScalar(other)
        elif isinstance(other,Leaf):
            if(other.getLayerSize()!=self.getLayerSize()):
                success = False
                Log.Error(LeafTag,"Leaf multiplication should have the same layers")
            i = 0
            for mat in self._matrixs:
                # todo why would it save time?
                #result += MM.sum(mat*other[i])
                result +=MM.sum( MM.matmulS(MM.transpose(mat),other[i]) )
                #result += np.sum(np.dot(np.transpose(mat),other[i]))
                i=i+1
        else:
            Log.Error(LeafTag,"unknown type for multiplication")
        if(success):
            return result
        else:
            return None
    
     # # the other Leaf should have the same laysize and same dimensions
   
    # for Leaf: other*self = self*other
    def __rmul__(self,other):
        return self*other

    def dot(self,other):
        success = True
        resultMats = []
        if isinstance(other,Leaf):
            if(other.getLayerSize()!=Leaf.getLayerSize()):
                success = False
                Log.Error(LeafTag,"Leaf multiplication should have the same layers")
            i = 0
            for mat in self._matrixs:
                # todo why would it save time? and make success
                resultMats.append(MM.matmulS(mat,other.getMatrix(i)))
                #resultMats.append(MM.dot(mat,other.getMatrix(i)))
                i=i+1
        if(success):
            return self._createInstance(resultMats)
        else:
            return None


    # def getMatrix(self,indexLayer):
    #     return self.__matrixs[indexLayer]
    def __getitem__(self, layerIndex):
        """Return the item at the given index."""
        return self._matrixs[layerIndex]

    def getLayerSize(self):
        return len(self._matrixs)
    
    # child class will stay child class
    def _createInstance(self, value):
        return self.__class__(value)
        
    def __repr__(self):
        strs = []
        for mat in self._matrixs:
            strs.append(str(mat))
        return " ".join(strs)
    
    def __str__(self):
        return repr(self)
    

      
        


        



                
                
                