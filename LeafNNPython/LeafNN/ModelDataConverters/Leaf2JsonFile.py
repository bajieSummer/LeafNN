from LeafNN.ModelDataConverters.BaseDataWriter import BaseDataWriter
from LeafNN.utils.Log import Log
from LeafNN.core.LeafModels.Leaf import Leaf
import json

TagLeaf2JsonFile = "Tag_Leaf_To_Json_File"
class Leaf2JsonFile(BaseDataWriter):

    def writeXY(self,leaf,filePath):
        Log.Error(TagLeaf2JsonFile,"Error: not implemented yet")

    def writeWB(self,leaf:Leaf,filePath):
        wb = []
        for i in range(leaf.getLayerSize()):
            mat = leaf[i]
            wb.append(mat.tolist())
        dict = {"WB":wb}
        with open(filePath, 'w') as json_file:
            json.dump(dict, json_file)


        