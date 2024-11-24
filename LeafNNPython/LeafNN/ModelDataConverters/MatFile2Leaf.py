from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.core.LeafModels.Leaf import Leaf
from LeafNN.core.LeafModels.NeuralLeaf import NeuralLeaf
from LeafNN.ModelDataConverters.BaseDataReader import BaseDataReader
from LeafNN.utils.Log import Log
import scipy.io
TagMatFile2Leaf = "Tag_MatFile_to_Leaf"
class MatFile2Leaf(BaseDataReader):
    def read_mat_file(self,file_path,variable_name=None):
        """
        Load data from a MATLAB MAT-file.

        Parameters:
        - file_path: str, the path to the .mat file.
        - variable_name: str, the name of the variable to extract (optional).
        
        Returns:
        - data: dict or numpy array, the loaded data.
        """
        try:
            # Load the MAT-file
            data = scipy.io.loadmat(file_path)

            # If a specific variable name is provided, return that variable
            if variable_name is not None:
                return data.get(variable_name, None)  # Return None if the variable is not found
            
            return data  # Return the entire data dictionary if no variable is specified

        except Exception as e:
            Log.Error(TagMatFile2Leaf,f"Error loading MAT-file: {e}")
            return None


    def readXYFromFile(self,filePath):
        data = self.read_mat_file(filePath)
        X = data.get('X',None)
        Y = data.get('y',None)
        if (X is None) and (Y is None):
            Log.Error(TagMatFile2Leaf,"can't find data from files")
            return None
        matArr = []
        matArr.append(X)
        matArr.append(Y)
        return Leaf(matArr)
    
    def readWBFromFile(self, filePath):
        #Log.Error(TagMatFile2Leaf,"Error: Not implemented")
        data = self.read_mat_file(filePath)
        matArr= []
        for key in data:
            matArr.append(data[key])
        return NeuralLeaf(matArr)
        #return None