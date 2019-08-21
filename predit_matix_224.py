# -*- coding: utf-8 -*-
import numpy as np
import xlrd
import pandas as pd
import openpyxl


        
workbook = xlrd.open_workbook("/cptjack/totem/yanyiting/Eight_classification/data/micoscopeInfo_slide.xlsx")
workbook = pd.read_excel("/cptjack/totem/yanyiting/Eight_classification/data/micoscopeInfo_slide.xlsx")
label_name = workbook[['name','slice']]

prediction = np.load('prediction_resnet224_list.npy')
arr= np.load('prediction_resnet224_list.npy')
pro = np.load('prediction_resnet224_list.npy')

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
            
            
from sklearn.metrics import classification_report
target_name = ['focus', 'focus1','focus2']
print(classification_report(labels, predictions, target_names = target_name))

arr = np.load('prediction_resnet224_list.npy')
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(labels,predictions)
plt.imshow(cm,interpolation='nearest',cmap = "Pastel1")
plt.title("Confusion matrix",size = 15)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks,["focus","focus1","focus2"],rotation = 45,size = 10)
plt.yticks(tick_marks,["focus","focus1","focus2"],size = 10)
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
        plt.annotate(str(np.round(cm[x][y]/a[x],3)),xy = (y,x),horizontalalignment="center",verticalalignment='center')
