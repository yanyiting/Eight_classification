# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:05:19 2018

@author: Biototem_1
"""
import pandas as pd
import numpy as np
import argparse
import datetime
#import GPUtil
import random
import keras
import glob
import time
import sys
import os

from keras.models import *
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D, Flatten
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.xception import Xception
from keras.initializers import Orthogonal
from keras.utils import to_categorical
from keras.preprocessing import image
from generators import DataGenerator
from scipy.misc import imresize
from keras.models import Model
from keras import backend as K
from keras import optimizers
from skimage import io
from keras.callbacks import CSVLogger, Callback, EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.callbacks import TensorBoard, ModelCheckpoint
import cv2
import numpy as np
from sklearn import cross_validation, metrics
import time
from sklearn.metrics import f1_score 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

top_model=load_model('/cptjack/totem/yanyiting/Eight_classification/code/Resnet50_gray_2class224/ResNet50_gray.hdf5')
#top_model.save('InceptionV3_complete.h5')
import cv2
import numpy as np
from sklearn import cross_validation, metrics

#test_folders = "./NCT/validation/"
test_folders = '/cptjack/totem/yanyiting/Eight_classification/gray/2_focus/val/'

batch_size_for_generators=32
nb_test_samples = sum([len(files) for root, dirs, files in os.walk(test_folders)])
test_steps = nb_test_samples//batch_size_for_generators


print("\nImages for Testing")
print("=" * 30)

img_width,img_height = 224, 224
test_datagen = DataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_folders,target_size=(img_width,img_height),batch_size=32,
                                                  class_mode='categorical',shuffle=False)

### 

test_loss, test_accuracy = top_model.evaluate_generator(test_generator,steps=test_steps,verbose=1)

predictions = top_model.predict_generator(test_generator,steps=test_steps,verbose=1)
prediction_list = np.argmax(predictions, axis=1)
labels = test_generator.classes[:len(predictions)]
test_generator.class_indices

print("\nTest Loss: %.3f" %(test_loss))
print("Test Accuracy: %.3f" %(test_accuracy))
print("=" * 30, "\n")

    
from sklearn.metrics import classification_report
target_name = ['focus', 'focus1']
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


#np.save("45_Xception_2_pro.npy", posteriors)
#np.save("27_validation", test_labels)
#np.save("45_Xception_2_pre.npy", predictions)

#plt.savefig('./hunxiao_matrix_2.tif',bbox_inches='tight')
#plt.show()