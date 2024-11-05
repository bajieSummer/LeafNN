
DEBUG_MODE = True
Is_INIT = False
#print('hello demoInit')
if DEBUG_MODE and not Is_INIT:
    #print('hello source debug')
    import os
    import sys
    # Get the current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #print(current_dir)
    # Add the root directory to the Python module search path
    root_dir = os.path.abspath(os.path.join(current_dir, "../"))
    #print(root_dir)
    sys.path.append(root_dir)
    Is_INIT = True


from LeafNN.utils.Log import Log
from LeafNN.utils.Log import LogOption
from LeafNN.utils.PathUtils import PathUtils
Log.config(PathUtils.getLogFolderPath(),LogOption())