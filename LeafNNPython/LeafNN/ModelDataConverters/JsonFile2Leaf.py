from LeafNN.ModelDataConverters.BaseDataReader import BaseDataReader
from LeafNN.utils.Log import Log
import json
from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.core.LeafModels.NeuralLeaf import NeuralLeaf 

TagJsonFile2Leaf="Tag_Json_File_To_Leaf"
class JsonFile2Leaf(BaseDataReader):
    
    def readXYFromFile(self,filePath):
        Log.Error(TagJsonFile2Leaf,"Error not implemented yet")

    def readWBFromFile(self,filePath):
        loaded_matrices_dict = None
        with open(filePath, 'r') as json_file:
            loaded_matrices_dict = json.load(json_file)
        MatricsList = loaded_matrices_dict.get("WB")
        leafMats = []
        for mat in MatricsList:
            leafMats.append(MM.array(mat))
        return NeuralLeaf(leafMats)





