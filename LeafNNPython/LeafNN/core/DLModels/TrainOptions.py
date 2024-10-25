'''
Author: Sophie
email: bajie615@126.com
Date: 2024-07-10 12:38:25
Description: file content
'''

class MonitorOption:
    def __init__(self):
        self.enable = False
        self.frequency = 1

class MonitorData:
    def __init__(self):
        self.grads = []
        self.costs = []
        self.rates = []
        self.iterationInds = []
        self.sucesses=[]

class TrainOption:
    def __init__(self):
        self.MaxIteration = 100
        self.MaxLineSearch = 20
        self.C1 = 0.01 # line Search: wolfe conditon 1
        self.C2 = 0.5  # line search: wolfe condtion 2
        self.INT = 0.1
        self.EXT = 3.0
        self.RATIO = 100
        self.monitorOption = MonitorOption()
        
from enum import Enum
class GradientOptions(Enum):
    BGD = 1
    SGD = 2
    MiniBGD = 3
    Momentum = 4
    Adam = 5
 


