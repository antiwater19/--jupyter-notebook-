#!/usr/bin/env python
# coding: utf-8

# In[21]:


import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# In[22]:


num_classes = 10


# In[23]:


(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()


# ### ResNet blocks

# In[24]:


def res_block(x, filters, stride = 1):
    shortcut = x

    x = layers.Conv2D(filters, 3, strides = stride, padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, strides = 1, padding = 'same')(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides = stride, padding = 'same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut]) # x.shape == shortcut.shape 이어야 함
    x = layers.ReLU()(x)

    return x


# In[25]:


def myModel(input_shape, filters, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    net = layers.Conv2D(filters, 3, 1, padding='same')(inputs)
    net = layers.BatchNormalization()(net)
    net = layers.ReLU()(net)

    net = res_block(net, 64)
    net = res_block(net, 64)
    net = res_block(net, 128, stride = 2) # half size
    net = res_block(net, 128)
    net = res_block(net, 256, stride = 2) # half size
    net = res_block(net, 256)

    net = layers.GlobalAveragePooling2D()(net)

    net = tf.keras.layers.Dense(num_classes, activation='softmax')(net)

    model = tf.keras.Model(inputs=inputs, outputs=net, name='test')

    return(model)


# In[26]:


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()


# In[27]:


x_train_4d = x_train / 255.0 # scaling to [0,1]
x_test_4d = x_test / 255.0 # scaling to [0,1]


# In[28]:


x_train_4d.shape, x_test_4d.shape


# In[29]:


input_shape = (32, 32, 3)
filters =32
model = myModel(input_shape, filters, num_classes)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy', 
              metrics=['accuracy'])


# In[ ]:


num_epochs = 10
bs = 64
hist = model.fit(x_train_4d, y_train, validation_data = (x_test_4d, y_test), batch_size=bs, shuffle=True, epochs=num_epochs)


# In[ ]:





# In[ ]:




