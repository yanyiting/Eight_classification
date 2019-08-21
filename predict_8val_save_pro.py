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
import xlrd
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES']='1'

workbook = xlrd.open_workbook("/cptjack/totem/yanyiting/Eight_classification/data/DatabaseInfo.xlsx")
workbook = pd.read_excel("/cptjack/totem/yanyiting/Eight_classification/data/DatabaseInfo.xlsx")
label_name = workbook[['Name','Slice #']]
prediction = np.load('prediction_resnet224_val.npy')
arr= np.load('prediction_resnet224_val.npy')
pro = np.load('prediction_resnet224_val.npy')

prediction_1 = sorted(prediction, key=lambda x: x[0])
label_name['Slice #']

clear = [8,9]
clear1=[7,10]
clear2=[6,11]
clear3=[5,12]
clear4=[4,13]
clear5=[3,14]
clear6=[2,15]
clear7=[1,16]
labels=[]
predictions = []
for i in range(len(label_name['Slice #'])):
    if prediction_1[i][0]==label_name['Name'][i]:
        if label_name['Slice #'][i] in clear:
            labels.append(0)
        elif label_name['Slice #'][i] in clear1:
            labels.append(1)
        elif label_name['Slice #'][i] in clear2:
            labels.append(2)
        elif label_name['Slice #'][i] in clear3:
            labels.append(3)
        elif label_name['Slice #'][i] in clear4:
            labels.append(4)
        elif label_name['Slice #'][i] in clear5:
            labels.append(5)
        elif label_name['Slice #'][i] in clear6:
            labels.append(6)
        elif label_name['Slice #'][i] in clear7:
            labels.append(7)
        predictions.append(int(prediction_1[i][1]))
    else:
        print('mis match')

predictions.count(0) 
       


Resnet_model = load_model('/cptjack/totem/yanyiting/Eight_classification/code/Resnet50_224/Resnet50.hdf5')

directory_name = '/cptjack/totem/yanyiting/Eight_classification/data/8_focus/val'

img_num = sorted(os.listdir(directory_name))
pro_list = []
prediction_list = []
font = cv2.FONT_HERSHEY_SIMPLEX
for k in range(len(img_num)):
    img = cv2.imread(directory_name+'/'+img_num[k])
    img_2 = img.copy()
#    img= cv2.resize(img(512,512))
    name=img
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img/255
    patch_img=[]
    patchs_pro = []
    for i in range(img.shape[1]//224):
        for j in range(img.shape[0]//224):
#               patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
            patch = img[224*i:224*(i+1),224*j:224*(j+1)]
            patch = cv2.resize(patch,(224,224))
            patch_img.append(patch)
            patch = np.expand_dims(patch, axis=0)
            patch_pro=Resnet_model.predict(patch)
            if labels[k]!= np.argmax(patch_pro):
                cv2.putText(img_2, str(np.round(patch_pro[0][0],2)), (224*i+112, 224*j+112), font, 0.5, (255,0,0),2)
            else:
                cv2.putText(img_2, str(np.round(patch_pro[0][0],2)), (224*i+112, 224*j+112), font, 0.5, (0,0,0),2)
            patchs_pro.append(patch_pro)
                
frame = pd.DataFrame(pro,index =name, columns = range(len(pro[0])))
frame.to_excel('val_pro_8.xlsx')
                
patch_img = np.array(patch_img)
Resnet_preds = Resnet_model.predict_classes(patch_img)
prediction= Counter(Resnet_preds).most_common(1)[0][0]
prediction_list.append((img_num[k].split(".")[0],prediction))
pro_list.append((img_num[k].split(".")[0],patchs_pro))
dir_name = '/cptjack/totem/yanyiting/Eight_classification/data/'
if not os.path.exists(dir_name): os.makedirs(dir_name)
cv2.imwrite(dir_name+'/'+img_num[k], img_2)
   



#np.save('prediction_best_list.npy', prediction_list)  
#np.save('pro_list.npy', pro_list)                     
                                      
                    

