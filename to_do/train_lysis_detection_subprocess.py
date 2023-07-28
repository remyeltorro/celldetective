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


from adccfactory.core.utils import *
from adccfactory.model_utils.lysis import LysisDetectionModel


import matplotlib.pyplot as plt
import seaborn as sns

# Allow memory growth for the GPU
try:
	physical_devices = list_physical_devices('GPU')
	set_memory_growth(physical_devices[0], True)
except:
	pass

model_folder = str(sys.argv[1])
instructions = np.load(model_folder+"/instructions.npy", allow_pickle=True).item()
print("Loaded instructions:",instructions)

pretrained_path = instructions["pretrained_path"]
target_directory = instructions["target_directory"]
model_signal_length = instructions["model_signal_length"]
model_name = instructions["model_name"]
data_folders = instructions["data_folders"]
recompile_pretrained = instructions["recompile_pretrained"]
augment = instructions["augment"]
batch_size = instructions["batch_size"]
epochs = instructions["epochs"]
learning_rate = instructions["learning_rate"]
channel_option = instructions["channel_option"]


model = LysisDetectionModel(pretrained=pretrained_path, 
							model_signal_length=model_signal_length,
							channel_option=channel_option,
							n_channels=len(channel_option)
							)

model.fit_from_directory(data_folders,
						model_name=model_name, 
						target_directory=target_directory,
						channel_option=channel_option,
						recompile_pretrained=recompile_pretrained,
						test_split=0.,
						augment=augment,
						epochs=epochs,
						learning_rate=learning_rate,
						batch_size=batch_size,
						)