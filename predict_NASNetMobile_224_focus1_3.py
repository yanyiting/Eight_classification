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

os.environ['CUDA_VISIBLE_DEVICES']='1'
NASNetMobile_model = load_model('/cptjack/totem/yanyiting/Eight_classification/code/NASNetMobile_/NASNetMobile_.hdf5')
directory_name = '/cptjack/totem/yanyiting/Eight_classification/data/png'


img_names = sorted(os.listdir(directory_name))


prediction_list = []
for k in range(len(img_names)):
    img = cv2.imread(directory_name+'/'+img_names[k])
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img/255
    patch_img = []
    for i in range(img.shape[0]//224):
        for j in range(img.shape[1]//224):
    #                patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
            patch = img[224*i:224*(i+1),224*j:224*(j+1)]
#            patch = cv2.resize(patch,(224,224))
            patch_img.append(patch)
             
    patch_img = np.array(patch_img)
    NASNetMobile_preds = NASNetMobile_model.predict_classes(patch_img)
    prediction= Counter(NASNetMobile_preds).most_common(1)[0][0]
    print(prediction)
    prediction_list.append((img_names[k].split(".")[0],prediction))
np.save('prediction_NASNetMobile224_list.npy', prediction_list)                    


#def read_directory(directory_name,resnet_model):
#    img_num = os.listdir(directory_name)
#    prediction_list = []
#    for k in range(len(img_num)):
#        img = cv2.imread(directory_name+'/'+img_num[k])
#        img=cv2.resize(img,(256,256))
#        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#        img = img/255
#        patch_img=[]
#        for i in range(img.shape[1]//224):
#            for j in range(img.shape[0]//224):
##                patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
#                patch = img[224*i:224*(i+1),224*j:224*(j+1)]
#                patch = cv2.resize(patch,(224,224))
#                patch_img.append(patch)
#             
#        patch_img = np.array(patch_img)
#        predictions = resnet_model.predict_classes(patch_img)
#        prediction= Counter(predictions).most_common(1)[0][0]
#        print(prediction)
#        prediction_list.append((img_num[k].split(".")[0],prediction))
#    return prediction_list
#        
#   
#directory_name = '/cptjack/totem/yanyiting/Eight_classification/data/png'
#prediction_list = read_directory(directory_name,resnet_model)
#np.save('prediction_resnet_list.npy', prediction_list)      
                                     
                    

