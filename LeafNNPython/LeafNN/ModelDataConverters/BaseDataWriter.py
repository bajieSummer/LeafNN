from abc import ABC, abstractmethod
class BaseDataWriter:
    def __init__(self):
       pass 
    @abstractmethod
    def writeXY(self,leaf):
        pass
    @abstractmethod
    def writeWB(self,leaf):
        pass

