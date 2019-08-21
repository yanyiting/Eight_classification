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

workbook = xlrd.open_workbook("/cptjack/totem/yanyiting/Eight_classification/data/micoscopeInfo_slide.xlsx")
workbook = pd.read_excel("/cptjack/totem/yanyiting/Eight_classification/data/micoscopeInfo_slide.xlsx")
label_name = workbook[['name','slice']]

prediction = np.load('/cptjack/totem/yanyiting/Eight_classification/code/prediction_resnet224_list.npy')
arr= np.load('/cptjack/totem/yanyiting/Eight_classification/code/prediction_resnet224_list.npy')
pro = np.load('/cptjack/totem/yanyiting/Eight_classification/code/prediction_resnet224_list.npy')

prediction_1 = sorted(prediction, key=lambda x: x[0])
label_name['slice']
#
focus = [8,9,7,10]
focus1=[6,11,5,12,4,13,3,14]
focus2=[2,15,1,16]
#focus=[1]
#focus1=[2]
#focus2=[3]

labels=[]
predictions = []
for i in range(len(label_name['slice'])):
    if prediction_1[i][0]==label_name['name'][i]:
        if label_name['slice'][i] in focus:
            labels.append(0)
        elif label_name['slice'][i] in focus1:
            labels.append(1)
        elif label_name['slice'][i] in focus2:
            labels.append(2)
        predictions.append(int(prediction_1[i][1]))
    else:
#        print('prediction_1[i][0]:',prediction_1[i][0])
#        print('label_name:',label_name['name'][i])
        print('miss match')      


Resnet50_model = load_model('/cptjack/totem/yanyiting/Eight_classification/code/Resnet50_224/ResNet.hdf5')

directory_name = '/cptjack/totem/yanyiting/Eight_classification/data/png'
def read_directory(directory_name,Resnet50_model,labels):
    img_num = sorted(os.listdir(directory_name))
    prediction_list = []
    pro_list = []
    patch_img = []
    for k in range(len(img_num)):
        img = cv2.imread(directory_name+'/'+img_num[k])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_2 = img.copy()
        img = img/255
        patch_img=np.array(patch_img)
        for i in range(img.shape[1]//224):
            for j in range(img.shape[0]//224):
#                patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
                patch = img[224*i:224*(i+1),224*j:224*(j+1)]
#                patch = cv2.resize(patch,(224,224))
        patch_img.append(patch)
           
        patch_img = np.array(patch_img)
        Resnet50_preds = Resnet50_model.predict(patch_img)
        prediction= Counter(Resnet50_preds).most_common(1)[0][0]
        print(Resnet50_preds) 
        if prediction!=labels[k]:
            label_name = '/cptjack/totem/yanyiting/Eight_classification/data/all_pro'+str(labels[k])
            if not os.path.exists(label_name):os.makedirs(label_name)
            cv2.imwrite(label_name+'/'+img_num[k], img_2)
        prediction_list.append((img_num[k].split(".")[0],prediction))
    return prediction_list

   
directory_name = '/cptjack/totem/yanyiting/Eight_classification/data/png'

prediction_list = read_directory(directory_name,Resnet50_preds, labels)
np.save('prediction_table.excel', prediction_list)  
