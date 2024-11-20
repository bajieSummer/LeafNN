from abc import ABC, abstractmethod

class BaseDataReader:
    
    def __init__(self):
       pass 
    
    @abstractmethod
    def readXYFromFile(self,filePath):
        pass

    @abstractmethod
    def readWBFromFile(self,filePath):
        pass
