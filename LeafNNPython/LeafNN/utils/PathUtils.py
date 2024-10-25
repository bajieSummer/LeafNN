'''
Author: Sophie
email: bajie615@126.com
Date: 2024-06-18 14:59:51
Description: file content
'''
import os

class PathUtils:
    project_rootpath = None
    demo_rootpath = None
    demo_datapath = None
    log_folderpath = None

    @staticmethod
    def getProjectRootPath():
        if PathUtils.project_rootpath is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            PathUtils.project_rootpath = os.path.abspath(os.path.join(current_dir, "../.."))
        return PathUtils.project_rootpath
    
    @staticmethod
    def getDemoRootPath():
        if PathUtils.demo_rootpath is None:
            PathUtils.demo_rootpath = os.path.join(PathUtils.getProjectRootPath(), "demos")
        return PathUtils.demo_rootpath
    
    @staticmethod
    def getDemoDatasPath():
        if PathUtils.demo_datapath is None:
            PathUtils.demo_datapath = os.path.join(PathUtils.getDemoRootPath(), "datas")
        return PathUtils.demo_datapath
    
    @staticmethod
    def getLogFolderPath():
        if PathUtils.log_folderpath is None:
            PathUtils.log_folderpath = os.path.join(PathUtils.getProjectRootPath(), "logs")
            if not os.path.exists(PathUtils.log_folderpath):
                # Create the parent directory
                os.makedirs(PathUtils.log_folderpath) 
        return PathUtils.log_folderpath

# P1 = PathUtils()   
# print(PathUtils.getDemoRootPath())


