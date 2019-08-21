# -*- coding: utf-8 -*-
import scipy.io as scio
from skimage import io
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
import glob
from keras.models import *
import time
import openslide as opsl
from collections import Counter
import numpy as np
import xlrd
import pandas as pd
import openpyxl


os.environ['CUDA_VISIBLE_DEVICES']='0'        
workbook = xlrd.open_workbook("/cptjack/totem/yanyiting/Eight_classification/data/micoscopeInfo_slide.xlsx")
workbook = pd.read_excel("/cptjack/totem/yanyiting/Eight_classification/data/micoscopeInfo_slide.xlsx")

label_name = workbook[['name','slice']]
#prediction = np.load('/cptjack/totem/yanyiting/Eight_classification/code/prediction_resnet224_2gray_mic.npy')
                      
#prediction_1 = sorted(prediction, key=lambda x: x[0])
label_name['slice']


focus = [1]
labels=[]
predictions = []
for i in range(len(label_name['slice'])):
    if label_name['slice'][i] in focus:
        labels.append(0)
    else:
        labels.append(1)



Resnet50_gray_model = load_model('/cptjack/totem/yanyiting/Eight_classification/code/Resnet50_gray_2class224/ResNet50_gray.hdf5')

directory_name = '/cptjack/totem/yanyiting/Eight_classification/data/png'

img_num = sorted(os.listdir(directory_name))
pro_list = []
prediction_list = []
name=[]
pro=[]
font = cv2.FONT_HERSHEY_SIMPLEX
for k in range(len(img_num)):
    print(k)
    
    img = cv2.imread(directory_name+'/'+img_num[k])
    img = np.array([img,img,img])
    img = img.transpose((1,2,0))
    img = np.expand_dims(img, axis=0)
    img_2 = img.copy()
#    img= cv2.resize(img(512,512))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img/255
    patch_img=[]
    patchs_pro = []
    for i in range(img.shape[1]//224):
        for j in range(img.shape[0]//224):
#               patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
            patch = img[224*i:224*(i+1),224*j:224*(j+1)]
#            patch = cv2.resize(patch,(224,224))
            patch_img.append(patch)
            patch = np.expand_dims(patch, axis=0)
            patch_pro=Resnet50_gray_model.predict(patch)
            if labels[k]!= np.argmax(patch_pro):
                cv2.putText(img_2, str(np.round(patch_pro[0][0],2)), (224*i+112, 224*j+112), font, 0.5, (255,0,0),2)
            else:
                cv2.putText(img_2, str(np.round(patch_pro[0][0],2)), (224*i+112, 224*j+112), font, 0.5, (0,0,0),2)
            patchs_pro.append(patch_pro)
                
frame = pd.DataFrame(pro,index =name, columns = range(len(pro[0])))
frame.to_excel('mic_resnet_gray_all_pro.xlsx')
                
patch_img = np.array(patch_img)
Resnet50_gray_preds = Resnet50_gray_model.predict_classes(patch_img)
prediction= Counter(Resnet50_gray_preds).most_common(1)[0][0]
prediction_list.append((img_num[k].split(".")[0],prediction))
pro_list.append((img_num[k].split(".")[0],patchs_pro))
dir_name = '/cptjack/totem/yanyiting/Eight_classification/data/mic_misss'
if not os.path.exists(dir_name): os.makedirs(dir_name)
cv2.imwrite(dir_name+'/'+img_num[k], img_2)
   



#np.save('prediction_best_list.npy', prediction_list)  
#np.save('pro_list.npy', pro_list)                     
                                      
                    
