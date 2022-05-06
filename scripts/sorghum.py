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
from keras_cv_attention_models import efficientnet
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

class SorghumCONFIG(object):
    TASK = "sorghum-id"
    TASK_ID = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

    MODEL = "EfficientNetB4"
    WEIGHTS = "imagenet"  # WEIGHTS = None
    OPTIMISER = "adam"

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

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.NUM_GPUS

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        self.on_change()

    def on_change(self):
        self.__dict__['BATCH_SIZE'] = self.IMAGES_PER_GPU * self.NUM_GPUS           # Very dirty

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

class CLAHE(ImageOnlyTransform):
    def apply(self, img, **params):
        return clahe(img)

class NORMALISE(ImageOnlyTransform):
    def apply(self, img, **params):
        return normalise(img)

def create_model():
# TODO:    
    tf.keras.backend.clear_session()

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():

        if CONFIG.MODEL == "EfficientNetB4":
            backbone = tf.keras.applications.EfficientNetB4(
                include_top=False, weights=CONFIG.WEIGHTS, input_shape=(CONFIG.IM_HEIGHT, CONFIG.IM_WIDTH, 3))

        elif CONFIG.MODEL == "EfficientNetB5":
            backbone = tf.keras.applications.EfficientNetB5(
                include_top=False, weights=CONFIG.WEIGHTS, input_shape=(CONFIG.IM_HEIGHT, CONFIG.IM_WIDTH, 3))
        
        elif CONFIG.MODEL == "EfficientNetB6":
            backbone = tf.keras.applications.EfficientNetB6(
                include_top=False, weights=CONFIG.WEIGHTS, input_shape=(CONFIG.IM_HEIGHT, CONFIG.IM_WIDTH, 3))

        elif CONFIG.MODEL == "EfficientNetB7":
            backbone = tf.keras.applications.EfficientNetB7(
                include_top=False, weights=CONFIG.WEIGHTS, input_shape=(CONFIG.IM_HEIGHT, CONFIG.IM_WIDTH, 3))

        elif CONFIG.MODEL == "EfficientNetV2S":
            backbone = tf.keras.applications.EfficientNetV2S(
                include_top=False, weights=CONFIG.WEIGHTS, input_shape=(CONFIG.IM_HEIGHT, CONFIG.IM_WIDTH, 3))

        elif CONFIG.MODEL == "EfficientNetV2M":
            backbone = tf.keras.applications.EfficientNetV2M(
                include_top=False, weights=CONFIG.WEIGHTS, input_shape=(CONFIG.IM_HEIGHT, CONFIG.IM_WIDTH, 3))

        elif CONFIG.MODEL == "EfficientNetV2L":
            backbone = tf.keras.applications.EfficientNetV2L(
                include_top=False, weights=CONFIG.WEIGHTS, input_shape=(CONFIG.IM_HEIGHT, CONFIG.IM_WIDTH, 3))

        #TODO: Add EffNetV2XL compatibility from keras_cv_attention_models
        elif CONFIG.MODEL == "EfficientNetV2XL":
            backbone = efficientnet.EfficientNetV2XL(include_preprocessing=False, num_classes=0) #norm layer and no top
        else:
            backbone = tf.keras.applications.EfficientNetB4(
                include_top=False, weights=CONFIG.WEIGHTS, input_shape=(CONFIG.IM_HEIGHT, CONFIG.IM_WIDTH, 3))

        backbone.trainable = True

        inputs = tf.keras.Input(shape=(CONFIG.IM_HEIGHT, CONFIG.IM_WIDTH, 3))
        x = backbone(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(CONFIG.NUM_CLASSES, activation="softmax", dtype='float32')(x)
        model = tf.keras.Model(inputs, outputs)

        if CONFIG.OPTIMISER == "adam" or "Adam":
            opt = tf.keras.optimizers.Adam(learning_rate=CONFIG.LR)
        elif CONFIG.OPTIMISER == "adamW" or "adamw":
            opt = tf.keras.optimizers.Adam(learning_rate=CONFIG.LR)
        elif CONFIG.OPTIMISER == "Radam" or "RAdam":
            opt = tfa.optimizers.RectifiedAdam(learning_rate=CONFIG.LR)
        elif CONFIG.OPTIMISER == "sgd" or "SGD":
            opt = tf.keras.optimizers.SGD(learning_rate=CONFIG.LR, momentum=0.9)
        elif CONFIG.OPTIMISER == "adabound" or "AdaBound":
            pass
        elif CONFIG.OPTIMISER == "adabelief" or "AdaBelief":
            opt = tfa.optimizers.AdaBelief(learning_rate=CONFIG.LR)  
        else:
            print("Unrecognised optimiser string in config, defaulting to Adam optimisation")
            opt = tf.keras.optimizers.Adam(learning_rate=CONFIG.LR)


        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=[tf.keras.metrics.CategoricalAccuracy(),
                            tf.keras.metrics.Precision(),
                            tf.keras.metrics.Recall(),
                            tfa.metrics.F1Score(average='macro', num_classes=CONFIG.NUM_CLASSES),
                            ])
        model.summary()
        return model

def load_data():
    image_df = pd.read_csv(PATH+'train_cultivar_mapping.csv')
    image_df.dropna(inplace=True)

    if DEBUG_FLAG:
        image_df = image_df.groupby('cultivar').apply(lambda x: x.sample(5))
        image_df.reset_index(drop=True)
        CONFIG.NUM_EPOCHS = 2
    image_df


    #TODO: Find "best" k-fold strategy 1-10
    kfold = StratifiedKFold(n_splits=CONFIG.KFOLDS, shuffle=True)

    for train_index, valid_index in kfold.split(image_df['image'], image_df['cultivar']):
        train_images, valid_images = image_df['image'].iloc[train_index], image_df['image'].iloc[valid_index]
        train_cultivar, valid_cultivar = image_df['cultivar'].iloc[
            train_index], image_df['cultivar'].iloc[valid_index]

    train_df = pd.DataFrame({'image': train_images, 'cultivar': train_cultivar})


    val_df = pd.DataFrame({'image': valid_images, 'cultivar': valid_cultivar})
    print('Number of Training Samples: ', len(train_df),
        '    Number of Validation Samples: ', len(val_df))

    return train_df, val_df

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

def augment_data(phase: str):
    if phase == "train":
        return Compose([
                CLAHE(p=1),
                A.RandomResizedCrop(height=CONFIG.IM_HEIGHT, width=CONFIG.IM_WIDTH),
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
                #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],p=1),
                
        ])
    else:
        return Compose([
                CLAHE(p=1),
                A.Resize(height=CONFIG.IM_HEIGHT, width=CONFIG.IM_WIDTH),
                #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],p=1),
        ])

#TODO: Add TTA with 90 rotations and random Rotation (+1.5% in tests)


def train_model():
    
    es = EarlyStopping(monitor = 'val_loss',
                       min_delta = CONFIG.EARLY_STOPPING_DELTA,
                       patience = CONFIG.EARLY_STOPPING_PATIENCE,
                       verbose = 1,
                       restore_best_weights = True)

    sv = ModelCheckpoint(MODEL_DIR + CONFIG.MODEL + '-{epoch:03d}-{val_loss:.4f}.h5',
                         monitor = 'val_loss',
                         verbose = 1,
                         save_best_only = False,
                         save_weights_only = False,
                         mode = 'min')

    sv_best = ModelCheckpoint(MODEL_DIR + CONFIG.MODEL + '-optimal.h5',
                              monitor = 'val_loss',
                              verbose = 1,
                              save_best_only = True,
                              save_weights_only = False,
                              mode = 'min')

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.4,
                                  verbose = 1,
                                  patience = CONFIG.STEPS_BEFORE_LR_DECAY,
                                  min_lr = CONFIG.LR_MIN)

    csv = CSVLogger('../history/' + CONFIG.MODEL + CONFIG.TASK_ID + '.csv')

    tb = TensorBoard(LOG_DIR, histogram_freq=1)



    model = create_model()
    train_df, val_df = load_data()

    train_datagen = ImageDataAugmentor(augment=augment_data("train"))
    val_datagen = ImageDataAugmentor(augment=augment_data("val"))

    train_augmented = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        shuffle=True,
        directory=TRAIN_DIR,
        x_col='image',
        y_col='cultivar',
        class_mode='categorical',
        target_size=(CONFIG.IM_HEIGHT, CONFIG.IM_WIDTH),
        batch_size=CONFIG.BATCH_SIZE)

    val_augmented = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        shuffle=True,
        directory=TRAIN_DIR,
        x_col='image',
        y_col='cultivar',
        class_mode='categorical',
        target_size=(CONFIG.IM_HEIGHT, CONFIG.IM_WIDTH),
        batch_size=CONFIG.BATCH_SIZE)

    CONFIG.NUM_CLASSES = len(train_augmented.class_indices)
    CONFIG.CLASS_INDICES = train_augmented.class_indices
    CONFIG.CLASS_ID, CONFIG.NUM_IMAGES = np.unique(train_augmented.classes, return_counts=True)
    CONFIG.MAX_VALUE = max(CONFIG.NUM_IMAGES)
    CONFIG.CLASS_WEIGHTS = {c: CONFIG.MAX_VALUE/n for c, n in zip(CONFIG.CLASS_ID, CONFIG.NUM_IMAGES)}

    CONFIG.STEP_SIZE_TRAIN = train_augmented.n//train_augmented.batch_size
    CONFIG.STEP_SIZE_VALID = val_augmented.n//val_augmented.batch_size

    CONFIG.display() 

    history = model.fit(train_augmented,
                        epochs=CONFIG.NUM_EPOCHS,
                        steps_per_epoch=CONFIG.STEP_SIZE_TRAIN,
                        callbacks=[es, tb, sv, sv_best, reduce_lr, csv],
                        verbose=2,
                        class_weight=CONFIG.CLASS_WEIGHTS,
                        validation_data=val_augmented,
                        validation_steps=CONFIG.STEP_SIZE_VALID)

def inference():
    tf.keras.backend.clear_session()
    sample_submission = pd.read_csv(PATH+'sample_submission.csv')

    if DEBUG_FLAG:
        sample_submission=sample_submission.head(100)

    test_gen = ImageDataAugmentor(augment=augment_data("test"))
    test_generator = test_gen.flow_from_dataframe(dataframe=sample_submission,
                                                directory=TEST_DIR,
                                                x_col='filename',
                                                y_col=None,
                                                target_size=(CONFIG.IM_HEIGHT, CONFIG.IM_WIDTH),
                                                color_mode='rgb',
                                                class_mode=None,
                                                batch_size=1,
                                                shuffle=False,)

    CONFIG.STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

    MODEL_PATH = MODEL_DIR + CONFIG.MODEL + '-optimal.h5'

    model = tf.keras.models.load_model(MODEL_PATH)

    test_generator.reset()
    results = model.predict(test_generator, verbose=2, steps=CONFIG.STEP_SIZE_TEST)

    predicted_class_indices = np.argmax(results, axis=1)

    labels = (CONFIG.CLASS_INDICES)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames = test_generator.filenames
    submission = pd.DataFrame({"Filename": [filename.replace('all_classes/', '')for filename in filenames],
                            "cultivar": predictions})

    submission_name = SAVE_DIR+'submission-'+CONFIG.MODEL + \
        '-'+str(CONFIG.BATCH_SIZE)+'-'+CONFIG.TASK_ID+'.csv'
    submission.to_csv(submission_name, index=False)

    if not DEBUG_FLAG:
        submit(submission_name, CONFIG)

def submit(submission, CONFIG):

    msg = CONFIG.MODEL+'-'+CONFIG.OPTIMISER+'-'+str(CONFIG.BATCH_SIZE)+'-'+CONFIG.TASK_ID
    os.system('kaggle competitions submit -c sorghum-id-fgvc-9 -f ' + submission+' -m "'+msg+'"')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Test Inputs from SBATCH.')
    parser.add_argument('--path', type=str, required=False,
                        help='test arg for printing')
    parser.add_argument('--batch', type=int, required=False,
                        help='batch size arg for printing')
    parser.add_argument('--model', type=str, required=False,
                        help='model arg for printing')
    parser.add_argument('--opt', type=str, required=False,
                        help='optimser arg for printing')
    parser.add_argument('--size', type=int, required=False,
                        help='image size arg for printing')
    parser.add_argument('--epochs', type=int, required=False,
                        help='epochs arg for printing')         
    parser.add_argument('--kfolds', type=int, required=False,
                        help='epochs arg for printing')  
    parser.add_argument('--lr', type=float, required=False,
                        help='starting lr size arg for printing')
    parser.add_argument('--lr_min', type=float, required=False,
                        help='minimum lr arg for printing')    
    parser.add_argument('--lr_steps', type=int, required=False,
                        help='image size arg for printing')
    parser.add_argument('--es_steps', type=int, required=False,
                        help='epochs arg for printing')  
    parser.add_argument('--es_delta', type=float, required=False,
                        help='epochs arg for printing')               

    args = parser.parse_args()
    CONFIG = SorghumCONFIG()


    if args.path is not None:
        PATH = args.path
        print("Path has been set (value is %s)" % args.path)

    if args.batch is not None:
        CONFIG.IMAGES_PER_GPU = args.batch
        print("Images per GPU has been set (value is %s)" % args.batch)

    if args.model is not None:
        CONFIG.MODEL = args.model
        print("Model has been set (value is %s)" % args.model)

    if args.opt is not None:
        CONFIG.OPTIMISER = args.opt
        print("Optimiser has been set (value is %s)" % args.opt)

    if args.size is not None:
        CONFIG.IM_HEIGHT = args.size
        CONFIG.IM_WIDTH = args.size
        print("Image Size has been set (value is %s)" % args.size)

    if args.epochs is not None:
        CONFIG.NUM_EPOCHS = args.epochs
        print("Num. Epochs has been set (value is %s)" % args.epochs)

    if args.kfolds is not None:
        CONFIG.KFOLDS = args.kfolds
        print("Num. K Folds has been set (value is %s)" % args.kfolds)

    if args.lr is not None:
        CONFIG.LR = args.lr
        print("Learning Rate has been set (value is %s)" % args.lr)

    if args.lr_min is not None:
        CONFIG.LR_MIN = args.lr_min
        print("Minimum Learning Rate has been set (value is %s)" % args.lr_min)

    if args.lr_steps is not None:
        CONFIG.STEPS_BEFORE_LR_DECAY = args.lr_steps
        print("Learning rate Decay Patience has been set (value is %s)" % args.lr_steps)

    if args.es_steps is not None:
        CONFIG.EARLY_STOPPING_PATIENCE = args.es_steps
        print("Early Stopping Patience has been set (value is %s)" % args.es_steps)

    if args.es_delta is not None:
        CONFIG.EARLY_STOPPING_DELTA = args.es_delta
        print("Early Stopping Delta has been set (value is %s)" % args.es_delta)
        
    CONFIG.display() 
    
    

    PATH = os.path.abspath(os.path.join(os.getcwd(), "../../../../datasets/sorghum-id-fgvc-9"))+"/"
    TRAIN_DIR = PATH+'train_images/'
    TEST_DIR = PATH+'test/'

    SAVE_DIR    = '../results/sorghum/'
    LOG_DIR     = "../logs/"+CONFIG.TASK+"-"+CONFIG.TASK_ID+'-'+CONFIG.MODEL+'-'+str(CONFIG.WEIGHTS)
    MODEL_DIR   = "../models/"+CONFIG.TASK+"-"+CONFIG.TASK_ID+'-'+CONFIG.MODEL+'-'+str(CONFIG.WEIGHTS)+'/'
    HIST_DIR    = "../history/"+CONFIG.TASK+"-"+CONFIG.TASK_ID+'-'+CONFIG.MODEL+'-'+str(CONFIG.WEIGHTS)+'/'
    SUB_DIR     = "../submissions/"+CONFIG.TASK+"-"+CONFIG.TASK_ID+'-'+CONFIG.MODEL+'-'+str(CONFIG.WEIGHTS)+'/'

    DEBUG_FLAG = False

    print(PATH)
    print(TRAIN_DIR)
    print(TEST_DIR)
    print(SAVE_DIR)
    print(LOG_DIR)
    print(MODEL_DIR)
    print(HIST_DIR)
    print(SUB_DIR)

    train_model()
    inference()
