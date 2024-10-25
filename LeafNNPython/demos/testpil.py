'''
Author: Sophie
email: bajie615@126.com
Date: 2024-06-18 14:28:57
Description: file content
'''
import demoInit
from PIL import Image
from PIL import ImageFilter
import os
from LeafNN.utils.PathUtils import PathUtils 
# # image read/show

drPath = os.path.join(PathUtils.getDemoRootPath(),"res")
filePath = drPath+"/Mars1.jpg"
print(filePath)
img = Image.open(filePath)
#img.show()


resized_img = img.resize((256, 256))
resize_path = drPath + "/resize_mars.jpg"

# Get the image format
img_format = img.format
# get the image size and shape
imgSize = img.size
print("the img size is ,{imgSize}");

# Print the image format
print(f"The image format is: {img_format}")
resized_img.save(resize_path)
resized_img.show()

# get greyscale img
greyscale = img.convert("L")
greyscale.save(drPath+"/greyscale.jpg")

# get color palette
colorP = img.convert("P",palette=Image.ADAPTIVE,colors=16)
colorP.show()
rgb_image = colorP.convert("RGB")
rgb_image.save(drPath+"/colorP.jpg")
#colorP.save(drPath+"/colorP.jpg")

# crop img
crop = resized_img.crop((32,32,128,128))
crop.save(drPath+"/crop.jpg")

blur = img.filter(ImageFilter.GaussianBlur(radius=1))
blur.show()
blur.save(drPath+"/blur.jpg")






