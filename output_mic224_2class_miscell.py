# -*- coding: utf-8 -*-
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
#prediction = np.load('prediction_resnet224_val.npy')
#arr= np.load('prediction_resnet224_val.npy')
#pro = np.load('prediction_resnet224_val.npy')                       
#prediction_1 = sorted (prediction, key=lambda x: x[0])

label_name['slice']


focus = [1]
unfocus=[2,3]
labels=[]
predictions = []
for i in range(len(label_name['slice'])):
    if label_name['slice'][i] in focus:
       labels.append(0)
    elif label_name['slice'][i] in unfocus:
       labels.append(1)
    else:
        print('miss match')




Resnet_model = load_model('/cptjack/totem/yanyiting/Eight_classification/code/Resnet50_gray_2class224/ResNet50_gray.hdf5')


def read_directory(directory_name,Resnet_model,labels):
    img_num = sorted(os.listdir(directory_name))
    prediction_list = []
    for k in range(len(img_num)):
        img = cv2.imread(directory_name+'/'+img_num[k])
        img= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.array([img,img,img])
        img = img.transpose((1,2,0))
        img = np.expand_dims(img, axis=0)
        img_2 = img.copy()
        img = img/255
        patch_img=[]
#        index_img = []
        for i in range(img.shape[1]//224):
            for j in range(img.shape[0]//224):
#                patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
                patch = img[224*i:224*(i+1),224*j:224*(j+1)]
                patch = cv2.resize(patch,(224,224))
                patch_img.append(patch)
             
        patch_img = np.array(patch_img)
        Resnet_preds = Resnet_model.predict_classes(patch_img)
        prediction= Counter(Resnet_preds).most_common(1)[0][0]
        if prediction!=labels[k]:
            label_name = '/cptjack/totem/yanyiting/Eight_classification/data/2class_misscell_mic/'+str(labels[k])
            if not os.path.exists(label_name):os.makedirs(label_name)
            cv2.imwrite(label_name+'/'+img_num[k], img_2)
        prediction_list.append((img_num[k].split(".")[0],prediction))
    return prediction_list
        
   
directory_name = '/cptjack/totem/yanyiting/Eight_classification/data/micro_png_224'

prediction_list = read_directory(directory_name,Resnet_model, labels)
np.save('Resnetpred_2class_misscell_mic.npy', prediction_list) 

