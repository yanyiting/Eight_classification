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
resnet_model = load_model('/cptjack/totem/yanyiting/Eight_classification/code/Resnet50_gray224/ResNet50_gray.hdf5')
directory_name = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/val'


focus_names = sorted(os.listdir(directory_name))

labels = []
prediction_list = []
pro_list = []
for k in range(len(focus_names)):
    print(k)
    img_names = sorted(os.listdir(directory_name+'/'+focus_names[0]))
    focus_list = []
    for l in range(len(img_names)):
        img_dir= cv2.imread(directory_name+'/'+focus_names[0]+'/'+img_names[0])
        
        img = cv2.cvtColor(img_dir,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(224,224))
        img = img/255
    patch_img = []
    for i in range(img.shape[0]//224):
        for j in range(img.shape[1]//224):
    #                patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
            patch = img[224*i:224*(i+1),224*j:224*(j+1)]
            patch = cv2.resize(patch,(224,224))
            patch_img.append(patch)
             
patch_img = np.array(patch_img)
img = np.expand_dims(img, axis=0)
resnet_preds = resnet_model.predict(img)
for i in range(len(resnet_preds[0])):
    resnet_preds[0][i] = resnet_preds[0][i]*(8-i)
            
#        prediction= Counter(resnet_preds).most_common(1)[0][0]
#        print(prediction)
    prediction = np.argmax(resnet_preds)
    prediction_list.append(prediction)
    pro_list.append(resnet_preds)
    labels.append(k)
    #prediction_list.append(focus_list)
np.save('prediction_resnet_8focus_gray_val.npy', prediction_list)
np.save('labels_resnet_8focus_gray_val.npy', labels)
np.save('pro_list_resnet_8focus_gray_val.npy', pro_list)                             


    
from sklearn.metrics import classification_report
target_name = ['focus', 'focus1','focus2','focus3','focus4','focus5','focus6','focus7']
print(classification_report(labels, prediction_list, target_names = target_name))

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(labels,prediction_list)
plt.imshow(cm,interpolation='nearest',cmap = "Pastel1")
plt.title("Confusion matrix",size = 15)
plt.colorbar()
tick_marks = np.arange(8)
plt.xticks(tick_marks,['focus', 'focus1','focus2','focus3','focus4','focus5','focus6','focus7'],rotation = 45,size = 10)
plt.yticks(tick_marks,['focus', 'focus1','focus2','focus3','focus4','focus5','focus6','focus7'],size = 10)
plt.tight_layout()
plt.ylabel("Actual label",size = 15)
plt.xlabel("Predicted",size =15)
width,height = cm.shape
a =[0,0,0,0,0,0,0,0]
for i in  range(8):
    for j in range(8):
        a[i] = cm[i][j]+a[i]
for x in range(width):
    for y in range(height):
        plt.annotate(str(np.round(cm[x][y]/a[x],2)),xy = (y,x),horizontalalignment="center",verticalalignment='center')

                    


