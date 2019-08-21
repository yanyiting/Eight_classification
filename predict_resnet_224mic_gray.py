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
resnet_model = load_model('/cptjack/totem/yanyiting/Eight_classification/code/Resnet50_gray_2class224/ResNet50_gray.hdf5')
directory_name = '/cptjack/totem/yanyiting/Eight_classification/data/micro_png_224'


focus_names = sorted(os.listdir(directory_name))

labels = []
prediction_list = []
pro_list = []
for k in range(len(focus_names)):
    print(k)
    img_names = sorted(os.listdir(directory_name+'/'+focus_names[k]))
    focus_list = []
    for l in range(len(img_names)):
        img= cv2.imread(directory_name+'/'+focus_names[k]+'/'+img_names[l],0)
        new_img = np.array([img,img,img])
        new_img = new_img.transpose((1,2,0))
        img = np.expand_dims(new_img, axis=0)

        img = img/255
        resnet_preds = resnet_model.predict(img)
        for i in range(len(resnet_preds[0])):
            resnet_preds[0][i] = resnet_preds[0][i]*(8-i)
#        prediction= Counter(resnet_preds).most_common(1)[0][0]
#        print(prediction)
        prediction = np.argmax(resnet_preds)
        prediction_list.append(prediction)
        pro_list.append(resnet_preds)
        if k==0:
            labels.append(0)
        else:
            labels.append(1)
    #prediction_list.append(focus_list)
np.save('prediction_resnet224_2gray_mic.npy', prediction_list)
np.save('labels_resnet224_2gray_mic.npy', labels)
np.save('pro_list_resnet224_2gray_mic.npy', pro_list)                             


    
from sklearn.metrics import classification_report
target_name = ['focus', 'unfocus']
print(classification_report(labels, prediction_list, target_names = target_name))

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(labels,prediction_list)
plt.imshow(cm,interpolation='nearest',cmap = "Pastel1")
plt.title("Confusion matrix",size = 15)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks,['focus', 'unfocus'],rotation = 45,size = 10)
plt.yticks(tick_marks,['focus', 'unfocus'],size = 10)
plt.tight_layout()
plt.ylabel("Actual label",size = 15)
plt.xlabel("Predicted",size =15)
width,height = cm.shape
a =[0,0]
for i in  range(2):
    for j in range(2):
        a[i] = cm[i][j]+a[i]
for x in range(width):
    for y in range(height):
        plt.annotate(str(np.round(cm[x][y]/a[x],2)),xy = (y,x),horizontalalignment="center",verticalalignment='center')

                    


