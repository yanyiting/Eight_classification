# -*- coding: utf-8 -*-
import numpy as np
import xlrd
import pandas as pd
import openpyxl
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'        
workbook = xlrd.open_workbook("/cptjack/totem/yanyiting/Eight_classification/data/micoscopeInfo.xlsx")
workbook = pd.read_excel("/cptjack/totem/yanyiting/Eight_classification/data/micoscopeInfo_slide.xlsx")
label_name = workbook[['name','slice']]

prediction = np.load('/cptjack/totem/yanyiting/Eight_classification/code/prediction_xception_64_list.npy')

prediction_1 = sorted(prediction, key=lambda x: x[0])
label_name['slice']



focus = [8,9]
focus1=[7,10,6,11,5,12,4,13,3,14,2,15,1,16]


labels=[]
predictions = []
for i in range(len(label_name['slice'])):
    if prediction_1[i][0]==label_name['name'][i]:
        if label_name['slice'][i] in focus:
            labels.append(0)
        elif label_name['slice'][i] in focus1:
            labels.append(1)
        predictions.append(int(prediction_1[i][1]))
    else:
        print('prediction_1[i][0]:',prediction_1[i][0])
        print('label_name:',label_name['name'][i])
        print('miss match')

predictions.count(0) 
       
#a = []    
#for i in range(len(predictions)):
#    if predictions[i] in [0,1]:
#        a.append(0)
#    elif predictions[i] in [2,3,4]:
#        a.append(1)
#    elif predictions[i] in [5,6,7]:
#        a.append(2)
        
from sklearn.metrics import classification_report
target_name = ['focus', 'focus1']
print(classification_report(labels, a, target_names = target_name))    
            