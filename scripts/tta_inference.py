import os
import sys
import math
import random
import datetime
import logging
import pickle

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np

import tensorflow_addons as tfa
from keras import Sequential, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard, ReduceLROnPlateau
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold

from PIL import Image as Img
import cv2 as cv

from ImageDataAugmentor.image_data_augmentor import *
from albumentations.core.composition import Compose, OneOf
from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A

devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
print("No. GPUs : "+str(len(devices)), "Tensorflow == "+tf.__version__, "Keras == "+tf.keras.__version__, sep='\n')


TASK = "sorghum-id"
TASK_ID = "2022-05-07-101114"

MODEL = "EfficientNetV2XL"
WEIGHTS = "imagenet"  # WEIGHTS = None
OPTIMISER = "adamW"
KFOLDS = "6fold"

IM_HEIGHT = 512
IM_WIDTH = 512
IM_CHANNELS = 3

NUM_CLASSES = 100
NUM_EPOCHS = 30
KFOLDS = 5
IMAGES_PER_GPU = 4
NUM_GPUS = len(devices)
BATCH_SIZE = IMAGES_PER_GPU*NUM_GPUS

STEPS_BEFORE_LR_DECAY = 2
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_DELTA = 0.01

LR = 1e-4
LR_MIN = 1e-8

f = open("../labels.pkl","rb")
labels = pickle.load(f)
f.close()

PATH = os.path.abspath(os.path.join(os.getcwd(), "../../../../datasets/sorghum-id-fgvc-9"))+"/"
TRAIN_DIR = PATH+'train_images/'
TEST_DIR = PATH+'test/'

SAVE_DIR    = '../results/sorghum/'
LOG_DIR     = "../logs/"+TASK+"-"+TASK_ID+'-'+MODEL+'-'+str(WEIGHTS)
MODEL_DIR   = "../models/"+TASK+"-"+TASK_ID+'-'+MODEL+'-'+str(WEIGHTS)+'/'
HIST_DIR    = "../history/"+TASK+"-"+TASK_ID+'-'+MODEL+'-'+str(WEIGHTS)+'/'
SUB_DIR     = "../submissions/"+TASK+"-"+TASK_ID+'-'+MODEL+'-'+str(WEIGHTS)+'/'

def clahe(img):
    c = cv.createCLAHE(clipLimit=40, tileGridSize=(16,16))  # create a clahe object
    t = np.asarray(img)                                     # convert to np array
    t = cv.cvtColor(t, cv.COLOR_BGR2HSV)                    # convert to OpenCV HSV
    t[:,:,-1] = c.apply(t[:,:,-1])                          # Apply CLAHE to the Value (greyscale) of the image
    t = cv.cvtColor(t, cv.COLOR_HSV2BGR)                    # Return to BGR OpenCV doamin
    t = Img.fromarray(t)                                    # Convert to PIL Image
    t = np.array(t)                                         # back to np array
    return t

def normalise(img):
    t = np.array(img,dtype=np.float32)/255
    return t

class CLAHE(ImageOnlyTransform):
    def apply(self, img, **params):
        return clahe(img)

class NORMALISE(ImageOnlyTransform):
    def apply(self, img, **params):
        return normalise(img)

TTA = [
        Compose([CLAHE(p=1),
                A.Resize(height=IM_HEIGHT, width=IM_WIDTH),
                A.Affine(rotate=0,p=1),]),

        Compose([CLAHE(p=1),
                A.Resize(height=IM_HEIGHT, width=IM_WIDTH),
                A.Affine(rotate=90,p=1),]),

        Compose([CLAHE(p=1),
                A.Resize(height=IM_HEIGHT, width=IM_WIDTH),
                A.Affine(rotate=180,p=2),]),
        
        Compose([CLAHE(p=1),
                A.Resize(height=IM_HEIGHT, width=IM_WIDTH),
                A.Affine(rotate=270,p=3),]),

        Compose([CLAHE(p=1),
                A.Resize(height=IM_HEIGHT, width=IM_WIDTH),
                A.HorizontalFlip(p=1),
                A.Affine(rotate=0,p=1),]),

        Compose([CLAHE(p=1),
                A.Resize(height=IM_HEIGHT, width=IM_WIDTH),
                A.HorizontalFlip(p=1),
                A.Affine(rotate=90,p=1),]),

        Compose([CLAHE(p=1),
                A.Resize(height=IM_HEIGHT, width=IM_WIDTH),
                A.HorizontalFlip(p=1),
                A.Affine(rotate=180,p=2),]),
        
        Compose([CLAHE(p=1),
                A.Resize(height=IM_HEIGHT, width=IM_WIDTH),
                A.HorizontalFlip(p=1),
                A.Affine(rotate=270,p=3),]),

        Compose([CLAHE(p=1),
                A.Resize(height=IM_HEIGHT, width=IM_WIDTH),
                A.ShiftScaleRotate(p=1),]),
]


tf.keras.backend.clear_session()
MODEL_DIR   = "../models/"+TASK+"-"+TASK_ID+'-'+MODEL+'-'+str(WEIGHTS)+'/'
MODEL_PATH = MODEL_DIR + MODEL + '-optimal.h5'
#MODEL_PATH = "sorghum-id-2022-04-30-124425-EfficientNetV2L-imagenet"
model = tf.keras.models.load_model(MODEL_PATH)

sample_submission = pd.read_csv(PATH+'sample_submission.csv')

results_aggregated = np.zeros([23639,NUM_CLASSES],dtype=np.float32)
results_collated = np.zeros([23639,NUM_CLASSES,len(TTA)],dtype=np.float32)

tta_df = pd.DataFrame(index=sample_submission.index)


for i, tta in enumerate(TTA):
    test_gen = ImageDataAugmentor(augment=tta)
    test_generator = test_gen.flow_from_dataframe(dataframe=sample_submission,
                                                directory=TEST_DIR,
                                                x_col='filename',
                                                y_col=None,
                                                target_size=(IM_HEIGHT, IM_WIDTH),
                                                class_mode=None,
                                                batch_size=1,
                                                shuffle=False,)

    STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
    test_generator.reset()
    results = model.predict(test_generator, verbose=2, steps=STEP_SIZE_TEST)
    
    results_collated[:,:,i] = results
    results_aggregated += results


results_aggregated = results_aggregated/len(TTA)
predicted_class_indices=np.argmax(results_aggregated,axis=1)

labels_dict = dict((v,k) for k,v in labels.items())
predictions = [labels_dict[k] for k in predicted_class_indices]

filenames = test_generator.filenames
submission = pd.DataFrame({"Filename": [filename.replace('all_classes/', '')for filename in filenames],
                        "cultivar": predictions})

submission_name = SAVE_DIR+'submission-TTA-avg-'+MODEL + \
    '-'+str(BATCH_SIZE)+'-'+TASK_ID+'.csv'
submission.to_csv(submission_name, index=False)