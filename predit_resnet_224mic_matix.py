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

os.environ['CUDA_VISIBLE_DEVICES']='0'
resnet_model = load_model('/cptjack/totem/yanyiting/Eight_classification/code/Resnet50_224/ResNet50.hdf5')
directory_name = '/cptjack/totem/yanyiting/Eight_classification/data/micro_png_224'


focus_names = sorted(os.listdir(directory_name))


labels=[0,1,2]
labels1 = [3,4,5]
labels2 = [6,7]
prediction_list = []
for k in range(len(focus_names)):
    print(k)
    img_names = sorted(os.listdir(directory_name+'/'+focus_names[k]))
    focus_list = []
    for l in range(len(img_names)):
        img_dir= cv2.imread(directory_name+'/'+focus_names[k]+'/'+img_names[l])
        
        img = cv2.cvtColor(img_dir,cv2.COLOR_BGR2RGB)
        img = img/255
#    patch_img = []
#    for i in range(img.shape[0]//224):
#        for j in range(img.shape[1]//224):
#    #                patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
#            patch = img[224*i:224*(i+1),224*j:224*(j+1)]
#            patch = cv2.resize(patch,(224,224))
#            patch_img.append(patch)
             
#    patch_img = np.array(patch_img)
        img = np.expand_dims(img, axis=0)
        resnet_preds = resnet_model.predict_classes(img)
        prediction= Counter(resnet_preds).most_common(1)[0][0]
        print(prediction)
        prediction_list.append(prediction)
        labels.append(k)
    #prediction_list.append(focus_list)
np.save('prediction_resnet224_mic.npy', prediction_list)
#np.save('labels.npy', labels)                       


#focus = [8,9,7,10]
#focus1=[6,11,5,12,4,13,3,14]
#focus2=[2,15,1,16]
#labels=[]
#predictions = []
#for i in range(len(label_name['slice'])):
#    if label_name['slice'][i] in focus:
#       labels.append(0)
#    elif label_name['slice'][i] in focus1:
#       labels.append(1)
#    elif label_name['slice'][i] in focus2:
#       labels.append(2)
#    else:
#        print('miss match')    
from sklearn.metrics import classification_report
target_name = ['focus', 'focus1','focus2']
print(classification_report(labels, prediction_list, target_names = target_name))

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(labels,prediction_list)
plt.imshow(cm,interpolation='nearest',cmap = "Pastel1")
plt.title("Confusion matrix",size = 15)
plt.colorbar()
tick_marks = np.arange(3)
plt.xticks(tick_marks,['focus', 'focus1','focus2'],rotation = 45,size = 10)
plt.yticks(tick_marks,['focus', 'focus1','focus2'],size = 10)
plt.tight_layout()
plt.ylabel("Actual label",size = 15)
plt.xlabel("Predicted",size =15)
width,height = cm.shape
a =[0,0,0]
for i in  range(3):
    for j in range(3):
        a[i] = cm[i][j]+a[i]
for x in range(width):
    for y in range(height):
        plt.annotate(str(np.round(cm[x][y]/a[x],2)),xy = (y,x),horizontalalignment="center",verticalalignment='center')

                    


