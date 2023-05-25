#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import cv2
import os
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn import metrics


# In[2]:


# os.mkdir('D:/MLCBP/images/F/grayscale')
# os.mkdir('D:/MLCBP/images/M/grayscale')
# os.mkdir('resizedF')
# os.mkdir('resizedM')
# os.mkdir('D:/MLCBP/bnwF')
# os.mkdir('D:/MLCBP/bnwM')


# GRAYSCALE

# In[3]:


import cv2
import glob
import os
images_path=glob.glob('D:/MLCBP/images/F/*.jpeg')
# os.mkdir('D:/MLCBP/images/F/grayscale')
i=0
for image in images_path:
    gray_images=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('D:/MLCBP/images/F/grayscale/image%02i.jpeg' %i, gray_images)
    i+=1


# In[4]:


import cv2
import glob
import os
images_path=glob.glob('D:/MLCBP/images/M/*.jpeg')
# os.mkdir('D:/MLCBP/images/M/grayscale')
i=0
for image in images_path:
    gray_images=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('D:/MLCBP/images/M/grayscale/image%02i.jpeg' %i, gray_images)
    i+=1


# RESIZING

# In[5]:


import cv2
import glob
import os
inputFolder='D:\MLCBP\images\F\grayscale'
# os.mkdir('resizedF')

i=0
for img in glob.glob(inputFolder + "/*.jpeg"):
    image=cv2.imread(img)
    imgResized=cv2.resize(image,(28,28))
    cv2.imwrite("resizedF/image%04i.jpeg" %i,imgResized)
    i+=1


# In[6]:


import cv2
import glob
import os
inputFolder='D:\MLCBP\images\M\grayscale'
# os.mkdir('resizedM')
i=0
for img in glob.glob(inputFolder + "/*.jpeg"):
    image=cv2.imread(img)
    imgResized=cv2.resize(image,(28,28))
    cv2.imwrite("resizedM/image%04i.jpeg" %i,imgResized)
    i+=1


# In[7]:


# flattening with normalization
def img_to_array(filename): 
    gray = cv2.imread(filename,cv2.IMREAD_GRAYSCALE) 
    flat_image = gray.flatten()/255
    return flat_image
folders=['resizedF','resizedM']
directory = 'D:\MLCBP'
image_arrays = []
for folder in folders:
    folder_path=os.path.join(directory,folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpeg') or filename.endswith('.png'):
            file_path=os.path.join(folder_path, filename)
            image_array = img_to_array(file_path)
            image_array=np.append(image_array,folders.index(folder))
            image_arrays.append(image_array)
df=pd.DataFrame(image_arrays)
df


# In[8]:


df.to_csv("D:\MLCBP\Dataset.csv",index=False)


# In[9]:


df=pd.read_csv("Dataset.csv")


# In[10]:


df.head(20)


# # Seperate Training & Testing

# In[11]:


# os.mkdir('D:/MLCBP/test data/Female/grayscale')
# os.mkdir('D:/MLCBP/test data/Male/grayscale')
# os.mkdir('D:/MLCBP/test data/Female/resizedF')
# os.mkdir('D:/MLCBP/test data/Male/resizedM')


# GRAYSCALE

# In[12]:


import cv2
import glob
import os
images_path=glob.glob('D:/MLCBP/test data/Female/*.jpeg')
# os.mkdir('D:/MLCBP/test data/Female/grayscale')
i=0
for image in images_path:
    gray_images=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('D:/MLCBP/test data/Female/grayscale/image%02i.jpeg' %i, gray_images)
    i+=1


# In[13]:


import cv2
import glob
import os
images_path=glob.glob('D:/MLCBP/test data/Male/*.jpeg')
# os.mkdir('D:/MLCBP/test data/Male/grayscale')
i=0
for image in images_path:
    gray_images=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('D:/MLCBP/test data/Male/grayscale/image%02i.jpeg' %i, gray_images)
    i+=1


# RESIZING

# In[14]:


import cv2
import glob
import os
inputFolder='D:/MLCBP/test data/Female/grayscale'
# os.mkdir('D:/MLCBP/test data/Female/resizedF')

i=0
for img in glob.glob(inputFolder + "/*.jpeg"):
    image=cv2.imread(img)
    imgResized=cv2.resize(image,(28,28))
    cv2.imwrite("D:/MLCBP/test data/Female/resizedF/image%04i.jpeg" %i,imgResized)
    i+=1


# In[15]:


import cv2
import glob
import os
inputFolder='D:/MLCBP/test data/Male/grayscale'
# os.mkdir('D:/MLCBP/test data/Male/resizedM')

i=0
for img in glob.glob(inputFolder + "/*.jpeg"):
    image=cv2.imread(img)
    imgResized=cv2.resize(image,(28,28))
    cv2.imwrite("D:/MLCBP/test data/Male/resizedM/image%04i.jpeg" %i,imgResized)
    i+=1


# In[16]:


# flattening with normalization
def img_to_array(filename): 
    gray = cv2.imread(filename,cv2.IMREAD_GRAYSCALE) 
    flat_image = gray.flatten()/255
    return flat_image
folders=['D:/MLCBP/test data/Female/resizedF','D:/MLCBP/test data/Male/resizedM']
directory = 'D:\MLCBP\test data'
image_arrays = []
for folder in folders:
    folder_path=os.path.join(directory,folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpeg') or filename.endswith('.png'):
            file_path=os.path.join(folder_path, filename)
            image_array = img_to_array(file_path)
            image_array=np.append(image_array,folders.index(folder))
            image_arrays.append(image_array)
d=pd.DataFrame(image_arrays)
d


# In[17]:


d.to_csv("D:\MLCBP\TestDataset.csv",index=False)


# In[ ]:




