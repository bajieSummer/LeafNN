import copy
from LeafNN.utils.Log import Log
MonitorTag="MonitorTag"

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

class TrainMonitor:
    def __init__(self):
        Log.Debug(MonitorTag,"TrainMonitorInit")
    
    def monitor(monitorOption:MonitorOption,resultMonitorData:MonitorData,costs,rates,grads):
        if(monitorOption and not monitorOption.enable):
            return 
        if(resultMonitorData == None):
            print("not inital MonitorData")
            return
        resultMonitorData.costs =copy.deepcopy(costs)
        resultMonitorData.rates = copy.deepcopy(rates)
        resultMonitorData.grads = copy.deepcopy(grads)
        for i in range(len(costs)):
            resultMonitorData.iterationInds.append(i)
