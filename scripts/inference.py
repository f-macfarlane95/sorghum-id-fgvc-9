import datetime, logging, os, sys, math, random
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv
from PIL import Image as Img

import pickle 
import albumentations as A

from ImageDataAugmentor.image_data_augmentor import *
from albumentations.core.composition import Compose, OneOf
from albumentations.core.transforms_interface import ImageOnlyTransform

devices = tf.config.list_physical_devices('GPU')
for device in devices:
   tf.config.experimental.set_memory_growth(device, True) 
print(devices)



TASK = "sorghum-id"
TASK_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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

def augment_data(phase: str):
    if phase == "train":
        return Compose([
                CLAHE(p=1),
                A.RandomResizedCrop(height=HEIGHT, width=WIDTH),
                A.Flip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.5),
                ], p=0.5),
                OneOf([
                    A.Blur(p=0.1),
                    A.GaussianBlur(p=0.1),
                    A.MotionBlur(p=0.1),
                ], p=0.1),
                OneOf([
                    A.GaussNoise(p=0.1),
                    A.ISONoise(p=0.1),
                    A.GridDropout(ratio=0.5, p=0.2),
                    A.CoarseDropout(max_holes=16, min_holes=8, max_height=16,
                                    max_width=16, min_height=8, min_width=8, p=0.2)
                ], p=0.2),
                NORMALISE(p=1),
                
        ])
    else:
        return Compose([
                CLAHE(p=1),
                A.Resize(height=HEIGHT, width=WIDTH),
                NORMALISE(p=1),
        ])


PATH = os.path.abspath(os.path.join(os.getcwd() ,"../../../../datasets/sorghum-id-fgvc-9"))+"/"

MODEL_NAME = "sorghum_id-20220426-013717-EfficientNetB7-imagenet" #### CHANGE ME
MODEL_PATH = "../models/"+MODEL_NAME+"/EfficientNetB7-optimal.h5"

test_dir = PATH+'test/'
save_dir    = '../results/sorghum/'

WIDTH = 512
HEIGHT = 512

f = open("../labels.pkl","rb")
labels = pickle.load(f)
f.close()

submission = pd.read_csv(PATH+'sample_submission.csv')

submission

test_gen = ImageDataAugmentor(augment=augment_data("test"))
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

reconstructed_model = tf.keras.models.load_model(MODEL_PATH)


test_generator.reset()
results = reconstructed_model.predict(test_generator,verbose=1,steps=STEP_SIZE_TEST)

predicted_class_indices=np.argmax(results,axis=1)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

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

submission_name = save_dir+'submission-EfficientNetB7-imagenet-20220426-013717.csv'
submission.to_csv(submission_name,index=False)
#os.system('kaggle competitions submit -c sorghum-id-fgvc-9 -f '+submission_name+' -m "EfficientNetB7 with CLAHE PreProc"')


# ---------------------------------------------------------------------------- #
#                                    MODEL2                                    #
# ---------------------------------------------------------------------------- #

MODEL_NAME = "sorghum_id-20220425-214757-EfficientNetB7-imagenet" #### CHANGE ME
MODEL_PATH = "../models/"+MODEL_NAME+"/EfficientNetB7-optimal.h5"

test_dir = PATH+'test/'
save_dir    = '../results/sorghum/'

WIDTH = 512
HEIGHT = 512

f = open("../labels.pkl","rb")
labels = pickle.load(f)
f.close()

submission = pd.read_csv(PATH+'sample_submission.csv')

submission

test_gen = ImageDataAugmentor(augment=augment_data("test"))
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

reconstructed_model = tf.keras.models.load_model(MODEL_PATH)


test_generator.reset()
results = reconstructed_model.predict(test_generator,verbose=1,steps=STEP_SIZE_TEST)

predicted_class_indices=np.argmax(results,axis=1)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

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

submission_name = save_dir+'submission-EfficientNetB7-imagenet-20220425-214757.csv'
submission.to_csv(submission_name,index=False)
#os.system('kaggle competitions submit -c sorghum-id-fgvc-9 -f '+submission_name+' -m "EfficientNetB7 with CLAHE PreProc"')

