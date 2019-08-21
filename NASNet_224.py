# -*- coding: utf-8 -*-

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
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, MaxPooling2D, Flatten,BatchNormalization
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications.nasnet import NASNetMobile
from keras.initializers import Orthogonal
from keras.utils import to_categorical
#from keras.preprocessing import image
#from generators import DataGenerator
from scipy.misc import imresize
from keras.models import Model
from keras import backend as K
from keras import optimizers
from skimage import io
from keras.callbacks import CSVLogger, Callback, EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from keras.callbacks import TensorBoard, ModelCheckpoint
#from keras.utils import multi_gpu_model
import time
from PIL import Image
#from keras.preprocessing.image import ImageDataGenerator
from generators import DataGenerator
#from keras.utils import multi_gpu_model



os.environ['CUDA_VISIBLE_DEVICES']='0'
model = NASNetMobile(weights = None,include_top = False,input_shape = (224,224,3))
#model.save('NASNetMobile(224).h5')
model.load_weights('/cptjack/sys_software_bak/keras_models/models/NASNet-mobile-no-top.h5')

top_model = Sequential()
top_model.add(model)
top_model.add(MaxPooling2D(pool_size=(2,2)))
top_model.add(Flatten())
top_model.add(Dropout(0.5))
#top_model.add(BatchNormalization())
top_model.add(Dense(128, activation='relu',kernel_initializer=Orthogonal()))
top_model.add(Dropout(0.5))
#top_model.add(BatchNormalization())
top_model.add(Dense(8, activation='softmax',kernel_initializer=Orthogonal()))
#parallel_model = multi_gpu_model(top_model,gpus=2)
#parallel_model.summary()
#top_model.load_weights('./Xception_decay.hdf5')

for layer in model.layers:
    layer.trainable = True

LearningRate = 0.01
decay = 0.0001
n_epochs = 30
sgd = optimizers.SGD(lr=LearningRate, decay=LearningRate/n_epochs, momentum=0.9, nesterov=True)
    
top_model.compile(optimizer = sgd,loss = 'categorical_crossentropy',metrics=['accuracy'])

trainable_params = int(np.sum([K.count_params(p) for p in set(top_model.trainable_weights)]))

non_trainable_params = int(np.sum([K.count_params(p) for p in set(top_model.non_trainable_weights)]))

print("\nModel Stats")
print("=" * 30)
print("Total Parameters: {:,}".format((trainable_params + non_trainable_params)))
print("Non-Trainable Parameters: {:,}".format(non_trainable_params))
print("Trainable Parameters: {:,}\n".format(trainable_params))


train_folders = '/cptjack/totem/yanyiting/Eight_classification/data/Eight_focus/train/'
validation_folders = '/cptjack/totem/yanyiting/Eight_classification/data/Eight_focus/val/'

img_width,img_height = 224,224
batch_size_for_generators = 32
train_datagen = DataGenerator(rescale = 1./255,rotation_range=178,horizontal_flip=True,vertical_flip=True,shear_range=0.6,fill_mode='nearest',stain_transformation = True)
train_generator = train_datagen.flow_from_directory(train_folders,target_size = (img_width,img_height),batch_size = 32,class_mode = 'categorical')
validation_datagen = DataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_directory(validation_folders,target_size=(img_width,img_height),
                                                              batch_size = 32,class_mode = 'categorical')

nb_train_samples = sum([len(files)for root,dirs,files in os.walk(train_folders)])
nb_validation_samples = sum([len(files)for root,dirs,files in os.walk(validation_folders)])




class Mycbk(ModelCheckpoint):
    def __init__(self, model, filepath ,monitor = 'val_loss',mode='min', save_best_only=True):
        self.single_model = model
        super(Mycbk,self).__init__(filepath, monitor, save_best_only, mode)
    def set_model(self,model):
        super(Mycbk,self).set_model(self.single_model)

def get_callbacks(filepath,model,patience=6):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = Mycbk(model, './NASNetMobile_/'+filepath)
    file_dir = './NASNetMobile_/log/'+ time.strftime('%Y_%m_%d',time.localtime(time.time()))
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    tb_log = TensorBoard(log_dir=file_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                  patience=2, verbose=0, mode='min', epsilon=-0.95, cooldown=0, min_lr=1e-8)
    log_cv = CSVLogger('./NASNetMobile_/' + time.strftime('%Y_%m_%d',time.localtime(time.time())) +'_log.csv', separator=',', append=True)
    return [es, msave,reduce_lr,tb_log,log_cv]

import time
file_path = "NASNetMobile_.hdf5"
callbacks_s = get_callbacks(file_path,top_model,patience=6)

batch_size_for_generators = 56
train_steps = nb_train_samples//batch_size_for_generators

valid_steps = nb_validation_samples//batch_size_for_generators

#start_time= time.time()
#top_model.save('Xception.hdf5')
top_model.fit_generator(generator=train_generator,epochs=n_epochs,steps_per_epoch=train_steps,validation_data=validation_generator,
                        validation_steps = valid_steps, callbacks=callbacks_s, verbose=1)
#cost_time=time.time()-start_time
#print("cost_time:",cost_time)
#top_model.save('Xception.hdf5')
