#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import tensorflow as tf
import os
import matplotlib.pyplot as plt

import cv2
import imghdr

import numpy as np


# In[96]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# ## GPU Configuration

# In[2]:


gpus = tf.config.experimental.list_physical_devices("GPU")
gpus


# In[3]:


for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# # Data Preparation

# In[4]:


data_path = r'D:\projects\z datasets\dogs-vs-cats\train'


# In[5]:


os.listdir(data_path)


# In[6]:


img_exts = ['jpeg', 'jpg', 'bmp', 'png']


# In[7]:


img_example = cv2.imread(os.path.join(data_path, 'cat', 'cat.11985.jpg'))
img_example.shape


# In[8]:


plt.imshow(cv2.cvtColor(img_example, cv2.COLOR_BGR2RGB))


# In[9]:


img_example = cv2.imread(os.path.join(data_path, 'cat', 'cat.10000.jpg'))
img_example.shape


# In[10]:


plt.imshow(cv2.cvtColor(img_example, cv2.COLOR_BGR2RGB))


# In[ ]:





# In[11]:


get_ipython().run_line_magic('pinfo2', 'tf.keras.utils.image_dataset_from_directory')


# tf.keras.utils.image_dataset_from_directory(
#     directory,
#     labels='inferred',
#     label_mode='int',
#     class_names=None,
#     color_mode='rgb',
#     batch_size=32,
#     image_size=(256, 256),
#     shuffle=True,
#     seed=None,
#     validation_split=None,
#     subset=None,
#     interpolation='bilinear',
#     follow_links=False,
#     crop_to_aspect_ratio=False,
#     **kwargs,
# )

# # Load Data

# In[12]:


# use tensorflow dataset pipeline to make dataset instead of loading everything into memory

data = tf.keras.utils.image_dataset_from_directory(data_path) # (data_path, batch_size=8, img_size= (128, 128))


# - tensor flow build a dataset of 24999 entries belonging to 2 classes

# In[13]:


type(data)


# In[14]:


data_iterator = data.as_numpy_iterator()


# In[15]:


batch = data_iterator.next()


# In[16]:


type(batch)


# In[17]:


len(batch) # imgs, labels


# In[18]:


batch[0].shape # [ 0 ] -> list of 32 imgs of size 256 * 256 * 3 


# In[19]:


batch[0][0].shape


# In[20]:


batch[1]


# In[21]:


batch[1].shape


# In[22]:


fig, ax= plt.subplots(ncols= 4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# In[23]:


other_batch = data_iterator.next()


# In[24]:


fig, ax= plt.subplots(ncols= 4, figsize=(20, 20))
for idx, img in enumerate(other_batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(other_batch[1][idx])


# # Preprocessing Images for DL

# - Normalize data

# In[25]:


data = data.map(lambda x,y: (x/255, y))


# In[26]:


scaled_data_iterator = data.as_numpy_iterator()


# In[27]:


scaled_batch = scaled_data_iterator.next()


# In[28]:


fig, ax= plt.subplots(ncols= 4, figsize=(20, 20))
for idx, img in enumerate(scaled_batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(scaled_batch[1][idx])


# split data

# - Split data into train, validation, and test sets

# In[29]:


train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)


# In[30]:


print(train_size, val_size, test_size) # train 547 batchs, val 156 batches, 78 test batchs


# In[31]:


# prepare data --> data is shuffled
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)


# In[32]:


len(test)


# # Createing DL Classifier

# Model 1

# In[34]:


base = Sequential ([
                        
    Conv2D(16, (3, 3), 1, activation= 'relu', input_shape= (256, 256, 3)),
    MaxPooling2D(),
    
    
    Conv2D(32, (3, 3), 1, activation= 'relu'),
    MaxPooling2D(),
    
    Conv2D(16, (3, 3), 1, activation= 'relu'),
    MaxPooling2D(),
    
    Flatten(),
    
    Dense(256, activation= 'relu'),
    Dense(1, activation= 'sigmoid')
                        
])


# In[35]:


base.compile('adam', loss= tf.losses.BinaryCrossentropy(), metrics= ['accuracy'])


# In[36]:


base.summary()


# In[37]:


logdir = r'D:\projects\z datasets\dogs-vs-cats\logs'


# In[38]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= logdir)
checkpoint = ModelCheckpoint('base_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)


# In[ ]:





# In[39]:


history = base.fit(train, epochs= 20, validation_data= val, callbacks= [tensorboard_callback, checkpoint])


# plot performance

# In[40]:


fig = plt.figure()
plt.plot(history.history['loss'], color= 'teal', label= 'loss')
plt.plot(history.history['val_loss'], color= 'orange', label= 'val_loss')
fig.suptitle('loss', fontsize= 15)
plt.legend(loc= 'upper left')
plt.show()


# In[ ]:





# In[41]:


fig = plt.figure()
plt.plot(history.history['accuracy'], color= 'teal', label= 'accuracy')
plt.plot(history.history['val_accuracy'], color= 'orange', label= 'val_accuracy')
fig.suptitle('loss', fontsize= 15)
plt.legend(loc= 'upper left')
plt.show()


# Model Is Overfitting

# In[ ]:





# - Another architecture

# In[ ]:





# In[ ]:





# In[ ]:





# In[104]:


model3 = Sequential ([
                        
    Conv2D(32, (3, 3), 1, activation= 'relu', kernel_regularizer=l2(0.01), input_shape= (256, 256, 3)),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.3),
    
    Conv2D(32, (3, 3), 1, activation= 'relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.3),
    
    Conv2D(64, (3, 3), 1, activation= 'relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.3),
    
    Flatten(),
    
    Dense(128, activation= 'relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(128, activation= 'relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(1, activation= 'sigmoid')
                        
])


# In[105]:


model3.compile('adam', loss= tf.losses.BinaryCrossentropy(), metrics= ['accuracy'])


# In[106]:


model3.summary()


# In[111]:


logdir = r'D:\projects\z datasets\dogs-vs-cats\logs'


# In[112]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= logdir)


# In[115]:


checkpoint = ModelCheckpoint('best_model2.h5', monitor='val_accuracy', mode= 'max', save_best_only= True, verbose= 1)


# In[116]:


history3 = model3.fit(train, epochs= 20, validation_data= val, callbacks= [tensorboard_callback])


# plot performance

# In[127]:


fig = plt.figure()
plt.plot(history3.history['loss'], color= 'teal', label= 'loss')
plt.plot(history3.history['val_loss'], color= 'orange', label= 'val_loss')
fig.suptitle('loss', fontsize= 15)
plt.legend(loc= 'upper left')
plt.xlim(0, 20)
plt.ylim(0, 1)
plt.show()


# In[ ]:





# In[126]:


fig = plt.figure()
plt.plot(history3.history['accuracy'], color= 'teal', label= 'accuracy')
plt.plot(history3.history['val_accuracy'], color= 'orange', label= 'val_accuracy')
fig.suptitle('loss', fontsize= 15)
plt.legend(loc= 'upper left')
plt.xlim(0, 20)
plt.ylim(0, 1)
plt.show()


# - Model Evaluation

# In[143]:


def evaluate_model(model, test_data):
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    accuracy = tf.keras.metrics.BinaryAccuracy()

    for X, y in test_data.as_numpy_iterator():
        yhat = model.predict(X)
        precision.update_state(y, yhat)
        recall.update_state(y, yhat)
        accuracy.update_state(y, yhat)

    print(f"Precision: {precision.result().numpy():.4f}")
    print(f"Recall: {recall.result().numpy():.4f}")
    print(f"Accuracy: {accuracy.result().numpy():.4f}")


# In[142]:


evaluate_model(model3, test)


# In[ ]:




