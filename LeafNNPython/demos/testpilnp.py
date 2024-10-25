'''
Author: Sophie
email: bajie615@126.com
Date: 2024-06-19 18:53:15
Description: file content
'''
import demoInit
from PIL import Image
import numpy as np
import os
from LeafNN.utils.PathUtils import PathUtils

resdir = os.path.join(PathUtils.getDemoRootPath(),"res/")
img = Image.open(resdir+"Mars1.jpg")
imgArr = np.array(img)
imgArr[:,:,2] = 0
print("imgArrShape->{}".format(imgArr.shape))
modImg = Image.fromarray(imgArr)
#modImg.show()
modImg.save(resdir+"modImg.jpg")
# crop from top, left, right,bottom
# center(512,512) l,t (512-256)-> (256,256), r,b(512+256,512+256) ->(768,768)
# top/bottom ->rows  left/right(colums)
# t/b-> 256->768,  l/r->256->768
cropArr = imgArr[256:768,256:768,:]
cropImg = Image.fromarray(cropArr)
cropImg.save(resdir+"crop.jpg")
cropImg.show()
print("cropArrShape->{}".format(cropArr.shape))


