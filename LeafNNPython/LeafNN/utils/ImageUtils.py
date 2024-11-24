from LeafNN.Bases.MathMatrix import MathMatrix as MM
from LeafNN.utils.Log import Log
import LeafNN.core.LeafModels.ModelData as MD
from PIL import Image
from pathlib import Path
import re
import os

Tag_Img_Utils = "Tag_Img_Utils"
class ImageUtils:
    def createGrayImgMatsFromData(X,ShowWidth=1024,isTranspose=False,pic_width=None):
        [m,n] = X.shape
        if pic_width is None:
            pic_width = int(MM.sqrt(n))
        pic_height = int(n/pic_width)
        Log.Debug(Tag_Img_Utils,f"X shape is{m} {n}, Width={pic_width},height={pic_height}")
        imgXs =[]
        #imgXs.append(X[0].reshape(pic_width,pic_height))
        columns = int(ShowWidth/pic_width)
        blanks = MM.ones([pic_width,pic_height])
        i = 0
        while i < m:
            imgXi = None
            j = 0
            while j < columns:
                if i >=m:
                    imgXi = MM.hstack([imgXi,blanks])
                else:
                    T = X[i].reshape(pic_width,pic_height)
                    if isTranspose:
                        T =  MM.transpose(T)
                    if imgXi is None:
                        imgXi = T
                    else:
                        imgXi = MM.hstack([imgXi,T])
                j+=1
                i+=1
            imgXs.append(imgXi)
        imgsMat = None
        for img in imgXs:
            if imgsMat is None:
                imgsMat = img
            else:
                imgsMat=MM.vstack([imgsMat,img])
        return imgsMat*255.0
        
    def displayImgsFromX(X,ShowWidth=1024,scale=1,isTranspose=False,imgWidth=None):
        imgMats = ImageUtils.createGrayImgMatsFromData(X,ShowWidth,isTranspose,imgWidth)
        img = Image.fromarray(imgMats)
        osize = img.size
        scaled_image = img.resize((osize[0]*scale,osize[1]*scale), Image.LANCZOS) 
        scaled_image.show()
        return scaled_image
    
    def createXYDataFromNumberPics(folderPath,isTranspose = False,isResize=False,picW=None,picH=None):
        directory = Path(folderPath)
        file_paths = list(directory.rglob('*.jpg')) + list(directory.rglob('*.png'))
        testX = None
        testY = None
        for fpath in file_paths:
            image = Image.open(fpath)
            if image is None:
                continue
            # if(isResize and picW and picH):
            #     image = image.resize((picW,picH))
            image = image.convert('L')
            filename = os.path.basename(fpath)
            numbers = re.findall(r'\d+', filename)
            number = None
            if numbers:
                number = float(numbers[0])
            else:
                continue
            image_array = MM.array(image)
            if isTranspose:
                image_array = MM.transpose(MM.array(image))
            Log.Debug(Tag_Img_Utils,f"img=\n{image_array}")
            image_array = image_array.flatten().reshape(1, -1)
            # to do why
        
            if(testX is None):
                testX = image_array
            else:
                testX = MM.vstack([testX,image_array])
            curY = MM.ones([1,1])*number
            if(testY is None):
                testY = curY
            else:
                testY = MM.vstack([testY,curY])
        return MD.ClassifyData(testX/255.0,testY)

    def saveNumberImgsFromXYData(folderPath,X,Y,picW,picH):
        [m,n] = X.shape
        # os.path.join(PathUtils.getDemoDatasPath()
        if(picW*picH)!=n:
            Log.Error(Tag_Img_Utils,"wrong picw,and pich")
            return
        for i in range(m):
            ix = X[i,:].reshape([picW,picH])*255.0
            #ix = ix.astype(np.uint8)
            img = Image.fromarray(ix)
            filePath = os.path.join(folderPath,f"Y{Y[i,0]}_x{i}.jpg")
            #img.save(filePath)
            #img.show()
            img_converted = img.convert('L') 
            #img_converted.show()
            img_converted.save(filePath)