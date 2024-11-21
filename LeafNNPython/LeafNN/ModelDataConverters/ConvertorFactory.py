import os
from LeafNN.utils.Log import Log
from LeafNN.ModelDataConverters.BaseDataReader import BaseDataReader
from LeafNN.ModelDataConverters.BaseDataWriter import BaseDataWriter
from LeafNN.ModelDataConverters.TxtFile2Leaf import TxtFile2Leaf
from LeafNN.ModelDataConverters.MatFile2Leaf import MatFile2Leaf
from LeafNN.ModelDataConverters.Leaf2JsonFile import Leaf2JsonFile
from LeafNN.ModelDataConverters.JsonFile2Leaf import JsonFile2Leaf
TagConvertorFactory = "TAG_ConvertorFactory"
class ConvertorFactory:
    __instance = None
    def __init__(self):
        self.file_reader_registry = {}
        self.file_writer_registry = {}
        self.registerDefaultReaders()
        self.registerDefaultWriters()

    def registerDefaultReaders(self):
        self.registerReader(".txt",TxtFile2Leaf())
        self.registerReader(".mat",MatFile2Leaf())
        self.registerReader(".json",JsonFile2Leaf())

    def registerDefaultWriters(self):
        self.registerWriter(".json",Leaf2JsonFile())

    def registerReader(self,extension:str,reader_class:type):
        self.file_reader_registry[extension] = reader_class
    
    def registerWriter(self,extension:str,writer_class:type):
        self.file_writer_registry[extension] = writer_class
    
    def getInstance():
        if(ConvertorFactory.__instance is None):
            ConvertorFactory.__instance = ConvertorFactory()
        return ConvertorFactory.__instance
    
    def readXYFromFile(self,filePath):
        _, extension = os.path.splitext(filePath)
        fileReader:BaseDataReader = self.file_reader_registry.get(extension,None)
        if(fileReader is None):
            Log.Error(TagConvertorFactory,f"Error: we don't support read file from this extension:{extension}")
            return None
        return fileReader.readXYFromFile(filePath)
    
    def readWB(self,filePath):
        _, extension = os.path.splitext(filePath)
        fileReader:BaseDataReader = self.file_reader_registry.get(extension,None)
        if(fileReader is None):
            Log.Error(TagConvertorFactory,f"Error: we don't support read file from this extension:{extension}")
            return None
        return fileReader.readWBFromFile(filePath)
    
    def writeWB(self,leaf,filePath):
        _, extension = os.path.splitext(filePath)
        fileWriter:BaseDataWriter = self.file_writer_registry.get(extension,None)
        if(fileWriter is None):
            Log.Error(TagConvertorFactory,f"Error: we don't support write file with this extension:{extension}")
            return
        return fileWriter.writeWB(leaf,filePath)

    
    