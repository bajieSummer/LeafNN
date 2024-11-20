from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.core.LeafModels.Leaf import Leaf
from LeafNN.ModelDataConverters.BaseDataReader import BaseDataReader
from LeafNN.utils.Log import Log

TagTexFile2Leaf = "Tag_txt_file_to_leaf"
class TxtFile2Leaf(BaseDataReader):

    def readXYFromFile(self,filePath):
        with open(filePath, 'r') as file:
            lineCount = 0
            data = []
            for line in file:
                elements = line.strip().split(',')  # Split the line by comma
                data.append([float(element) for element in elements])
                lineCount += 1
        # transform into np.array
        result = MM.array(data)
        [n,m] = result.shape
        matArr = []
        matArr.append(result[:,0:m-1])
        matArr.append(result[:,m-1:m])
        return Leaf(matArr)

    def readWBFromFile(self, filePath):
        Log.Error(TagTexFile2Leaf,"Error: Not implemented")
        return None
        
    

    
        
