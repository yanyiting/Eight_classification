# -*- coding: utf-8 -*-
#one
img_path = '/cptjack/totem/yanyiting/Eight_classification/data/micro_png_224/focus2/slide03_strip02_slice03_position03_5_8.png'
from keras.preprocessing import image
import numpy as np
img = image.load_img(img_path,target_size=(224,224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor,axis =0)
img_tensor /= 255.
print(img_tensor.shape)
#test image
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()
#model
from keras import models
model=Resnet50_224
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(input=model.input,outputs = layer_outputs)
#predict model
activations = activation_model.predict(img_tensor)
first_layer_activation = activation[0]
print(first_layer_activation.shape)
# 4
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0,:,:,4],cmap='viridis')
#7
plt.matshow(first_layer_activation[0,:,:,7],cmap='viridis')
#
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
    
images_per_row = 16
for layer_name,layer_activation in zip(layer_names,activations):
    n_features = layer_activation.shape[-1]
    
    size=layer_activation.shape[1]
    n_cols = n_features //images_per_row
    display_grid = np.zeros(size*n_cols,images_per_row*size)
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :,:,
                                             col*images_per_row +row]
            channel_image -=channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image +=128
            channel_image = np.clip(channel_image,0,255).astype('uint8')
            display_grid[col*size:(col+1)*size,
                         row*size:(row+1)*size] = channel_image
            scale = 1./size
plt.figure(figsize = (scale *display_grid.shape[1],
                      scale*display_grid.shape[0]))
plt.title(layer_name)
plt.grid(False)
plt.imshow(display_grid,aspect='auto',cmap='viridis')
    
# tensor
from keras.applications import Resnet50
from keras import backed as K
model = Resnet50(weights = 'imagenet',
                 include_top = False)
layer_name = 'block3_conv1'
filter_index = 0
layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:,:,:,filter_index])
#grad
grads =K.gradients(loss,model.input)[0]

#numpy
iterate = K.function([model.input],[loss,grads])
import numpy as np
loss_value,grads_value = iterate([np.zeros((1,224,224,3))])
#random
input_img_data = np.random.random((1,224,224,3))*20+128.
step = 1.
for i in range(40):
    loss_value,grads_value = iterate([input_img_data])
    input_img_data +=grads_value *step
    
#tensor 
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std()+1e-5)
    x *=0.1
    
    x +=0.5
    x = np.clip(x,0,1)
    x *=255
    x = np.clip(x,0,255).astype('uint8')
    return x
# patten
def generate_pattern(layer_name,filter_index,size = 224):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])
    
    grads = K.gradients(loss,model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads)))+1e-5)
    iterate = K.function([model.input],[loss,grads])
    input_img_data = np.random.random((1,size,size,3))*20+128.
    step =1.
    for i in range(40):
        loss_value,grads_value = iterate([input_img_data])
        input_img_data += grads_value *step
        
    img = input_img_data[0]
    return deprocess_image(img)
plt.imshow(generate_pattern('block3_conv1',0))
#network
layer_name ='block1_conv1'
size = 64
margin = 5
results= np.zeros((8*size+7*margin,8*size+7*margin,3))
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name,i+(j*8),size=size)
        horizontal_start = i *size+i*margin
        horizontal_end = horizontal_start+size
        vertical_start = j*size+j*margin
        vertical_end = vertical_start +size
        results[horizontal_start:horizontal_end,
               vertical_start:vertical_end,:]=filter_img
plt.figure(figsize=(20,20))
plt.imshow(results)

#resnet
from keras.applications.resnet50 import resnet50
model = resnet50(weight='imgenet')

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input,decode_predictions
import numpy as np
img_path = '/cptjack/totem/yanyiting/Eight_classification/data/micro_png_224/focus2/slide03_strip02_slice03_position03_5_8.png'
img = image.load_img(img_path,target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims (x,axis =0)
x = preprocess_input(x)
preds = model.predict(x)
print('Predicted:',decode_predictions(preds,top=3)[0])

focus1_output = model.output[:,386]
last_lonv_layer = model.get_layer('block5_conv3')
grads = K.gradients(focus1_output,last_conv_layer.output)[0]
pooled_grads=K.mean(grads,axis=(0,1,2))
iterate = K.function([model.input],
                     [pooled_grads,last_conv_layer.output[0]])
pooled_grads_value,conv_layer_output_value = iterate(x)
for i in range(224):
    conv_layer_output_value[:,:,i]*=pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value,axis =-1)

#heatmap
heatmap = np.maximum(heatmap,0)
heatmap /=np.max(heatmap)
plt.matshow(heatmap)

#crop
import cv2
img= cv2.imread(img_path)
heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_PNG)
superiposed_img = heatmap *0.4 +img
cv2.imwrite('/cptjack/totem/yanyiting/Eight_classification/data/focus1.png',superimposed_img)



