#!/usr/bin/env python
# coding: utf-8

# In[1]:


conda install -c conda-forge opencv


# In[2]:


conda install -c conda-forge r-sys


# In[3]:


conda install -c jmcmurray os


# In[1]:


import os 


# In[10]:


# Importing OpenCV package
import cv2


# In[11]:


# Reading the image
img = cv2.imread( r"C:\Users\91831\Desktop\pho.jpg")


# In[12]:


# Converting image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[13]:


# Loading the required haar-cascade xml classifier file
faceCascade = cv2.CascadeClassifier(r"C:\Users\91831\Desktop\haarcascade_frontalface_default.xml")


# In[14]:


# Applying the face detection method on the grayscale image
faces = faceCascade.detectMultiScale(
    gray_img,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)


# In[15]:


print("[INFO] Found {0} Faces.".format(len(faces)))


# In[16]:


# Iterating through rectangles of detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_color = img[y:y + h, x:x + w]
    print("[INFO] Object found. Saving locally.")
    cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)


# In[17]:


status = cv2.imwrite('fin.jpg', img)
print("[INFO] Image faces_detected.jpg written to filesystem: ", status)


# In[ ]:





# In[ ]:





# In[ ]:




