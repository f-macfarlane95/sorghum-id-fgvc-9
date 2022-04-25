import datetime, logging, os, sys, math, random
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import albumentations as alb

from ImageDataAugmentor.image_data_augmentor import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from albumentations.core.composition import Compose, OneOf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

devices = tf.config.list_physical_devices('GPU')
for device in devices:
   tf.config.experimental.set_memory_growth(device, True) 
print(devices)

TASK = "sorghum-id"
TASK_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
PATH = os.path.abspath(os.path.join(os.getcwd() ,"../../../../datasets/sorghum-id-fgvc-9"))+"/"

train_dir = PATH+'train_images/'
test_dir = PATH+'test/'

save_dir    = '../results/sorghum/'
log_dir     = os.path.join("../logs/",''.join([TASK,"-",TASK_ID]))
model_dir   = os.path.join("../models/",''.join([TASK,"-",TASK_ID]))+'/'

WIDTH = 512
HEIGHT = 512

submission = pd.read_csv(PATH+'sample_submission.csv')
submission

test_gen= ImageDataAugmentor()
test_generator = test_gen.flow_from_dataframe(dataframe=submission,
                                              directory=test_dir,
                                              x_col='filename',
                                              y_col=None,
                                              target_size=(WIDTH,HEIGHT),
                                              color_mode='rgb',
                                              class_mode=None,
                                              batch_size=1,
                                              shuffle=False,)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
STEP_SIZE_TEST,test_generator.n,test_generator.batch_size

MODEL_PATH = "../models/sorghum-id-20220422-174202-EfficientNetB4-imagenet/EfficientNetB4-optimal.h5"

reconstructed_model = tf.keras.models.load_model(MODEL_PATH)

import pickle 

predicted_class_indices=np.argmax(results,axis=1)

f = open("../labels.pkl","rb")
labels = pickle.load(f)
f.close()

labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
submission=pd.DataFrame({"Filename":[filename.replace('all_classes/','')for filename in filenames],
                      "cultivar":predictions})
submission

submission_name = '../results/submission_EfficientnetB4ImageNet'+TASK_ID+'.csv'
submission.to_csv(submission_name,index=False)

#!kaggle competitions submit -c sorghum-id-fgvc-9 -f $submission_name -m "With flow from dataframe generator"
submission_name = "../results/sorghum/submission-EfficientNetB4-None-20220420-205414.csv"
os.system('kaggle competitions submit -c sorghum-id-fgvc-9 -f '+submission_name+' -m "os.system test"')

