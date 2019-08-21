# -*- coding: utf-8 -*-
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
import xlrd
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES']='1'

workbook = xlrd.open_workbook("/cptjack/totem/yanyiting/Eight_classification/data/DatabaseInfo.xlsx")
workbook = pd.read_excel("/cptjack/totem/yanyiting/Eight_classification/data/DatabaseInfo.xlsx")
label_name = workbook[['Name','Slice #']]

label_name['Slice #']
focus = [8,9]
focus1=[7,10]
focus2=[6,11]
focus3=[5,12]
focus4=[4,13]
focus5=[3,14]
focus6 =[2,15]
focus7=[1,16]
labels=[]
predictions = []
for i in range(len(label_name['Slice #'])):
    if label_name['Slice #'][i] in focus:
        labels.append(0)
    elif label_name['Slice #'][i] in focus1:
        labels.append(1)
    elif label_name['Slice #'][i] in focus2:
        labels.append(2)
    elif label_name['Slice #'][i] in focus3:
        labels.append(3)
    elif label_name['Slice #'][i] in focus4:
        labels.append(4)
    elif label_name['Slice #'][i] in focus5:
        labels.append(5)
    elif label_name['Slice #'][i] in focus6:
        labels.append(6)
    elif label_name['Slice #'][i] in focus7:
        labels.append(7)
Resnet_model = load_model('/cptjack/totem/yanyiting/Eight_classification/code/Resnet50_224/Resnet50.hdf5'
directory_name ='/cptjack/totem/yanyiting/Eight_classification/data/Eight_focus/val'
img_num = sorted(os.listdir(directory_name))
pro_list = []
font = cv2.FONT_HERSHEY_SIMPLEX
for k in range(len(img_num)):
    img = cv2.imread(directory_name+'/'+img_num[k])
    img_2 = img.copy()
#    img= cv2.resize(img(512,512))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img/255
    patchs_pro = []
    for i in range(img.shape[1]//224):
        for j in range(img.shape[0]//224):
#               patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
            patch = img[224*i:224*(i+1),224*j:224*(j+1)]
            patch = cv2.resize(patch,(224,224))
            patch = np.expand_dims(patch, axis=0)
            patch_pro=Xception_model.predict(patch)
            patchs_pro.append(round(patch_pro[0][0],2))
    pro_list.append(patchs_pro)

         
frame = pd.DataFrame(pro_list,index =label_name['Name'], columns = range(len(pro_list[0])))
frame.to_excel('all_focus.xlsx')
                

                    


                    

