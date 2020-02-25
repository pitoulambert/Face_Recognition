
# coding: utf-8

# # Face Classification 

# Objective:
# 
# In this assignment, we will do a CNN classification based on images captured from human faces of various angles.
# 
# 
# As mentioned in the dataset, there are 10 target categories available in the dataset.
# 
# 1. Arjun Prasad
# 2. Ashutosh Kumar
# 3. K.Baskaran
# 4. L.Sriniveau
# 5. Raj Singh
# 6. Rajesh Babu
# 7. RK Thakur
# 8. Sakthi Saravanan
# 9. Shaik Anwar
# 10. SS Muthu

# ##### Step 1: Importing libraries

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import load_files # used to load files in a directory structure
from keras.utils import np_utils
from glob import glob #used for wild card characters in between file paths
from keras.preprocessing import image #for image preprocessing                  
from tqdm import tqdm # to maintain a progress bar
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import time
from keras.callbacks import ModelCheckpoint # to save best model weights while training  
import matplotlib.pyplot as plt # for plotting
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import model_from_json # to save whole model & its weight to disk
import cv2 # to display images in final function
from keras.preprocessing.image import ImageDataGenerator


# ##### Step 2: Loading the dataset

# In[2]:


def load_dataset(path):
    data=load_files(path)
    face_files=np.array(data['filenames'])
    face_target=np_utils.to_categorical(np.array(data['target']),10)
    return face_files, face_target

print('Loading Train Files and Targets')
train_files, train_target = load_dataset(r"E:\face_classification_dataset\train")
print('Loading Complete!')
print('There are %d training face images.' % len(train_files))

#list of plant names
face_names= [item[10:-1] for item in sorted(glob(r"E:\face_classification_dataset\train\*"))]
print('There are %d total face categories.' % len(face_names))


# ##### Step 3: Preprocess the data
'''
When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

(nb_samples,rows,columns,channels),
 
where nb_samples corresponds to the total number of images (or samples), and rows, columns, and channels correspond to the number of rows, columns, and channels for each image, respectively.

The path_to_tensor function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN. The function first loads the image and resizes it to a square image that is  224×224  pixels. Next, the image is converted to an array, which is then resized to a 4D tensor. In this case, since we are working with color images, each image has three channels. Likewise, since we are processing a single image (or sample), the returned tensor will always have shape

(1,224,224,3).
 
The paths_to_tensor function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape

(nb_samples,224,224,3).
 
Here, nb_samples is the number of samples, or number of images, in the supplied array of image paths. It is best to think of nb_samples as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!
'''
# In[5]:


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# In[6]:


# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255


# ##### Step 4: Split the train set
We have loaded the data and converted into desired form. Now, to check our model performances, lets split our train set into train and validation sets. Validation set is used to validate the model performance
# In[8]:


from sklearn.model_selection import train_test_split # used to split our data to train and validation sets

# do not change seed to reproduce my results
seed = 123
np.random.seed(seed)



# Split the train and the validation set
train_tensors, val_train, train_target, val_targets = train_test_split(train_tensors,
                                              train_target, 
                                              test_size=0.05,
                                              random_state=seed
                                             )

print(train_tensors.shape)
print(val_train.shape)
print(train_target.shape)
print(val_targets.shape)


# ##### Step 5: Build the CNN Architecture

# In[9]:


model = Sequential()

model.add(Conv2D(input_shape=(224,224,3),filters=16,kernel_size=2,activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32,kernel_size=2,activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64,kernel_size=2,activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128,kernel_size=2,activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(GlobalAveragePooling2D())

model.add(Dense(10,activation='softmax'))

model.summary()


# In[10]:


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[11]:


epochs = 15

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.model.hdf5', 
                               verbose=1, save_best_only=True)
start=time.time()

history= model.fit(train_tensors, train_target, 
          validation_data=(val_train, val_targets),
          epochs=epochs, batch_size=32,callbacks=[checkpointer], verbose=1)

end=time.time()

total_time=end-start
print("Time Taken(in Minutes) to Train the Model:", total_time/60)


# In[44]:


# Save model.
model.save('saved_models/weights.best.model.hdf5')
print( history['val_acc'][-1], history['val_loss'][-1] )


# In[45]:


# Accuracy
model.load_weights('saved_models/weights.best.model.hdf5')
accuracy=model.evaluate(val_train,val_targets,batch_size=32)
print('Accuracy of a Model: ',accuracy[-1])


# In[48]:


model_json = model.to_json()
with open("saved_models/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("saved_models/model.hdf5")
print("Saved model to disk")


# ##### Step 6: Recognition of human faces

# In[49]:


#This Function takes an image (path) as an input and return the class of that image

def face_classification(img_path):
    json_file = open('saved_models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved_models/model.hdf5")
    print("Loaded model from disk")
    print("Preprocessing Image")
    img_tensors = path_to_tensor(img_path).astype('float32')/255
    img = cv2.imread(img_path)
    cv_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgplot = plt.imshow(cv_color)
    pred=loaded_model.predict_classes(img_tensors)
    print('Predicted Class of the faces:',pred)


# In[50]:


face_classification(r'E:\face_classification_dataset\test\Raj Singh\e418bd71-6260-4f34-a78b-c97e3bc565e0.jpg')
