# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)
std,mean=100,15
x = std+mean*np.random.randn(10000)
n,bins,patchse = plt.hist(x,50,density=True,facecolor ='g',alpha = 0.75)

plt.xlabel('')
plt.ylabel('Probability')
plt.title('Std,Mean')
plt.text(60,.025,r'$\std=100,\\mean=15$')
plt.xlim(40,160)
plt.ylim(0,0.03)
plt.grid(True)
plt.show()

