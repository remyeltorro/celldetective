#!/usr/bin/python

import sys
import os
import sys
import gc

import time
import shutil

import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model, clone_model
from tensorflow.config.experimental import list_physical_devices, set_memory_growth
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision

from sklearn.metrics import confusion_matrix, classification_report
from stardist.models import StarDist2D
from stardist.models import Config2D, StarDist2D, StarDistData2D

from adccfactory.core.utils import *

import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
from scipy.ndimage import shift


# Allow memory growth for the GPU
try:
	physical_devices = list_physical_devices('GPU')
	set_memory_growth(physical_devices[0], True)
except:
	pass

model_folder = str(sys.argv[1])
print(model_folder)


model = StarDist2D(None, name=os.path.split(model_folder)[-1], basedir=os.path.split(model_folder)[0])

print("Model basedir: ",model.basedir)
print("Model logdir: ",model.logdir)

# Load data
instructions = np.load(model_folder+"/training_instructions.npy",allow_pickle=True)

x_train = instructions.item().get("x_train")
x_test = instructions.item().get("x_test")

y_train = instructions.item().get("y_train")
y_test = instructions.item().get("y_test")

bs = instructions.item().get("batch_size")
n_epochs = instructions.item().get("n_epochs")
from_scratch = instructions.item().get("from_scratch")
augmentation = instructions.item().get("augmentation")

model.config.train_epochs = n_epochs
model.config.train_batch_size = bs
model.config.train_learning_rate = 0.0003
model.config.train_reduce_lr = {'factor': 0.1, 'patience': 30, 'min_delta': 0.000001}

print(model.config)

if from_scratch:
	print(os.path.split(model_folder)[0]+"/"+os.path.split(model_folder)[-1]+"/config.json")
	os.remove(os.path.split(model_folder)[0]+"/"+os.path.split(model_folder)[-1]+"/config.json")
	conf = Config2D(
	    n_rays       = 32,
	    grid         = (2,2),
	    use_gpu      = True,
	    n_channel_in = 2,
	    unet_dropout = 0.1,
	    train_learning_rate = 0.0003,
	    unet_n_conv_per_depth = 3,
	    train_reduce_lr = {'factor': 0.1, 'patience': 100, 'min_delta': 0.000001},
	    train_epochs = n_epochs,
	    train_batch_size = bs,
	)
	print(conf)
	model =  StarDist2D(conf, name=os.path.split(model_folder)[-1], basedir=os.path.split(model_folder)[0])


del instructions
gc.collect()

if augmentation:

	def random_fliprot(img, mask): 
	    assert img.ndim >= mask.ndim
	    axes = tuple(range(mask.ndim))
	    perm = tuple(np.random.permutation(axes))
	    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
	    mask = mask.transpose(perm) 
	    for ax in axes: 
	        if np.random.rand() > 0.5:
	            img = np.flip(img, axis=ax)
	            mask = np.flip(mask, axis=ax)
	    return img, mask 

	def random_intensity_change(img):
	    img = img*np.random.uniform(0.8,2) + np.random.uniform(-0.2,0.2)
	    return img

	def random_shift(image,mask):
	    
	    input_shape = image.shape[0]
	    max_shift = input_shape*0.1 #0.05
	    
	    shift_value_x = random.choice(np.arange(max_shift))
	    if tf.random.uniform(()) > 0.5:
	        shift_value_x*=-1

	    shift_value_y = random.choice(np.arange(max_shift))
	    if tf.random.uniform(()) > 0.5:
	        shift_value_y*=-1
	    
	    image = shift(image,[shift_value_x,shift_value_y,0],order=3,mode="constant",cval=0.0)
	    mask = shift(mask,[shift_value_x,shift_value_y],order=0,mode="constant",cval=0.0)
	    
	    return image,mask

	def augmenter(x, y):
	    """Augmentation of a single input/label image pair.
	    x is an input image
	    y is the corresponding ground-truth label image
	    """
	    x, y = random_fliprot(x, y)
	    x = random_intensity_change(x)
	    # add some gaussian noise
	    sig = 0.005*np.random.uniform(0,1)
	    x = x + sig*np.random.normal(0,1,x.shape)
	    x,y = random_shift(x,y)
	    return x, y	

	model.train(x_train, y_train, validation_data=(x_test,y_test), augmenter=augmenter)

else:

	model.train(x_train, y_train, validation_data=(x_test,y_test))

model.optimize_thresholds(x_train+x_test,y_train+y_test)

del model
gc.collect()

