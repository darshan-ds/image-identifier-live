from django.shortcuts import render
# Create your views here.

from django.core.files.storage import FileSystemStorage

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import json
from tensorflow import Graph
from tensorflow.compat.v1 import Session
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions




img_height, img_width=224,224


model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model=load_model('./models/MobileNetV2Model.h5')



def index(request):
    context={'a':1}
    return render(request,'index.html',context)



def predictImage(request):
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x=x/255
    x=x.reshape(1,img_height, img_width,3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)

    import numpy as np
    predictedLabel = decode_predictions(predi, top=1)[0][0][1]

    context={'filePathName':filePathName,'predictedLabel':predictedLabel}
    return render(request,'results.html',context) 

def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath, 'listOfImages':listOfImages}
    return render(request,'viewDB.html',context) 