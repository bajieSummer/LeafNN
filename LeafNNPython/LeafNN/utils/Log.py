from enum import Enum
from datetime import datetime
import logging
import os

# class LogLevel(Enum):
#     Debug = 0b0001
#     Info = 0b0010
#     Warning = 0b0100
#     Error = 0b1000

class LogTag:
    DLModels="DLModels"
    Utils = "Utils"

class LogLevel(Enum):
    Debug = logging.DEBUG
    Info = logging.INFO
    Warning = logging.WARNING
    Error = logging.ERROR
    Critical = logging.CRITICAL

# LogLevel2Name ={
#     LogLevel.Debug:"D",
#     LogLevel.Info:"I",
#     LogLevel.Warning:"W",
#     LogLevel.Error:"E",
#     LogLevel.Critical:"C"
# }

class LogOption:
    def __init__(self):
        self.fileMinLevel = LogLevel.Debug
        self.consoleMinLevel = LogLevel.Debug
        self.loggerName = "DLLog"
        self.enableConsole = True
    
class Log:
    __shareLog = None
    def __init__(self, folderPath,logOption:LogOption):
        self.folderPath = folderPath
        now = datetime.now()
        self.fileName = now.strftime("%Y_%m_%d_%H_%M_%S")  # e.g., 22/10/2023
        self.logger = logging.getLogger(logOption.loggerName)
        self.logger.setLevel(LogLevel.Debug.value)
        self.fullPath = os.path.join(folderPath, self.fileName)
        #formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(message)s')
        # Create file handler
        if not os.path.exists(self.fullPath):
            with open(self.fullPath, 'a'):
                pass
        file_handler = logging.FileHandler(self.fullPath)
        file_handler.setLevel(logOption.fileMinLevel.value)  # Log errors and above to file
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Create console handler
        if(logOption.enableConsole):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logOption.consoleMinLevel.value)  # Log debug and above to console
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
    def logging(self,logLevel:LogLevel,logTag,msg):
        currentTime = datetime.now()
        #msg=f"[{currentTime}]:[{logLevel.name}[{logTag}]{msg}]"
        msg=f"[[{logLevel.name}]:[{logTag}]{msg}]"
        self.logger.log(logLevel.value,msg)

    def config(folderPath,logOption:LogOption):
        if Log.__shareLog is None:
            Log.__shareLog = Log(folderPath,logOption)
        # todo when __shareLog is not none, 
    
    def log(logLevel:LogLevel,logTag,msg):
        if Log.__shareLog is None:
            print("LogError: please call config before use this log function")
            return
        Log.__shareLog.logging(logLevel,logTag,msg)

    def Debug(logTag,msg):
        Log.log(LogLevel.Debug,logTag,msg)
    
    def Info(logTag,msg):
        Log.log(LogLevel.Info,logTag,msg)

    def Warning(logTag,msg):
        Log.log(LogLevel.Warning,logTag,msg)

    def Error(logTag,msg):
        Log.log(LogLevel.Error,logTag,msg)
    
    def Critical(logTag,msg):
        Log.log(LogLevel.Critical,logTag,msg)
    
        
        
    


