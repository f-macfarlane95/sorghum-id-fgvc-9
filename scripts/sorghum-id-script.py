# ---------------------------------------------------------------------------- #
#                           SORGHUM ID CLASSIFICATION                          #
# ---------------------------------------------------------------------------- #

# ---------------------------------- IMPORTS --------------------------------- #

# TODO: Add if name is main and input args

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image as Img
import cv2 as cv
from sklearn.model_selection import StratifiedKFold
from ImageDataAugmentor.image_data_augmentor import *
from albumentations.core.composition import Compose, OneOf
import albumentations as A
from keras import Sequential, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import logging
import os
import sys
import math
import random

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


DEBUG_FLAG = False
devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
    print(devices, tf.__version__, tf.keras.__version__,sep='\n')

PATH = os.path.abspath(os.path.join(
    os.getcwd(), "../../../../datasets/sorghum-id-fgvc-9"))+"/"

TRAIN_DIR = PATH+'train_images/'
TEST_DIR = PATH+'test/'


# ---------------------------- VARIABLE DEFINITION --------------------------- #
# TODO: TASK_ID import SLURM ID in sbatch
# TODO: BASE_BATCH_SIZE import BASE_BATCH_SIZE ID in sbatch
class CONFIG:  # Default Values overwritten by batch files input

    TASK = "sorghum-id"
    TASK_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    MODEL = "EfficientNetB7"
    WEIGHTS = "imagenet"  # WEIGHTS = None

    NUM_GPUS = len(devices)
    BASE_BATCH_SIZE = 8
    BATCH_SIZE = BASE_BATCH_SIZE*NUM_GPUS
    EPOCH = 1
    KFOLDS = 5

    LR = 1e-4
    LR_MIN = 1e-8

    HISTEQ = "CLAHE"  # CLAHE, NONE

    IM_WIDTH = 512
    IM_HEIGHT = 512

    SAVE_DIR = '../results/sorghum/'
    LOG_DIR = "../logs/"+TASK+"-"+TASK_ID+'-'+MODEL+'-'+str(WEIGHTS)
    MODEL_DIR = "../models/"+TASK+"-"+TASK_ID+'-'+MODEL+'-'+str(WEIGHTS)+'/'


if __name__ == "__main__":

    # -------------------- READ TRAINING IMAGES INTO DATAFIELD ------------------- #

    image_df = pd.read_csv(PATH+'train_cultivar_mapping.csv')
    image_df.dropna(inplace=True)

    if DEBUG_FLAG:
        image_df = image_df[:2000]
        CONFIG.EPOCH = 1

    # ----------------------------- K-FOLD VALIDATION ---------------------------- #

    kfold = StratifiedKFold(n_splits=CONFIG.KFOLDS, shuffle=True)

    for train_index, valid_index in kfold.split(image_df['image'], image_df['cultivar']):
        train_images, valid_images = image_df['image'].iloc[train_index], image_df['image'].iloc[valid_index]
        train_cultivar, valid_cultivar = image_df['cultivar'].iloc[
            train_index], image_df['cultivar'].iloc[valid_index]

    train_df = pd.DataFrame({'image': train_images, 'cultivar': train_cultivar})


    val_df = pd.DataFrame({'image': valid_images, 'cultivar': valid_cultivar})
    print('Number of Training Samples: ', len(train_df),
        '    Number of Validation Samples: ', len(val_df))

    # clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(8,8))
    # def claheImage(img):
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     gray = gray.astype(np.uint16)
    #     eq = clahe.apply(gray)
    #     eq = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    #     eq = eq.astype(np.float32)
    #     return eq


    def CLAHE(img):
        clahe = cv.createCLAHE(clipLimit=40, tileGridSize=(10, 10))
        t = np.asarray(img)
        t = cv.cvtColor(t, cv.COLOR_BGR2HSV)
        t[:, :, -1] = clahe.apply(t[:, :, -1])
        t = cv.cvtColor(t, cv.COLOR_HSV2BGR)
        t = Img.fromarray(t)
        t = np.array(t)
        return t


    # ---------------------------- IMAGE AUGMENTATION ---------------------------- #
    # TODO: ADD PREPROCESSING (CLAHE)
    def augments(phase: str):
        if phase == "train":
            return Compose([
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
                # A.Normalize(
                #         mean=[0.485, 0.456, 0.406],
                #         std=[0.229, 0.224, 0.225],
                # ),
            ])
        else:
            return Compose([
                A.RandomResizedCrop(height=CONFIG.IM_HEIGHT, width=CONFIG.IM_WIDTH),
                # A.Normalize(
                #     mean=[0.485, 0.456, 0.406],
                #     std=[0.229, 0.224, 0.225],
                # ),
            ])

    train_datagen = ImageDataAugmentor(augment=augments("train"), preprocess_input=CLAHE)
    val_datagen = ImageDataAugmentor(augment=augments("val"),preprocess_input=CLAHE)

    # ------------------------- READ IMAGES TO GENERATORS ------------------------ #

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

    num_classes = len(train_augmented.class_indices)
    class_id, num_images = np.unique(train_augmented.classes, return_counts=True)
    max_value = max(num_images)
    class_weights = {c: max_value/n for c, n in zip(class_id, num_images)}

    # ---------------------------------------------------------------------------- #
    #                                   TRAINING                                   #
    # ---------------------------------------------------------------------------- #

    # ------------------------- MODEL FUNCTION DEFINITION ------------------------ #


    def create_model():

        tf.keras.backend.clear_session()

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            backbone = tf.keras.applications.EfficientNetB7(
                include_top=False, weights=CONFIG.WEIGHTS, input_shape=(CONFIG.IM_HEIGHT, CONFIG.IM_WIDTH, 3))
            backbone.trainable = True
            model = Sequential([
                # TODO
                tf.keras.Input(shape=(CONFIG.IM_HEIGHT, CONFIG.IM_WIDTH, 3)),
                backbone,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation='softmax')
            ])

            opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
            model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=[tf.keras.metrics.CategoricalAccuracy(),
                                tf.keras.metrics.Precision(),
                                tf.keras.metrics.Recall(),
                                # DEBUG:
                                tfa.metrics.F1Score(
                                    average='macro', num_classes=num_classes),
                                ])
        model.summary()
        return model


    model = create_model()

    # --------------------------------- CALLBACKS -------------------------------- #

    es = EarlyStopping(monitor='val_categorical_accuracy',
                    min_delta=0.01,
                    patience=10,
                    verbose=1,
                    restore_best_weights=True)

    # cp = ModelCheckpoint(SAVE_DIR + MODEL +'-{epoch:04d}-{val_loss:.4f}-{val_accuracy:.4f}.ckpt',
    #                      monitor='val_loss',
    #                      verbose=1,
    #                      save_best_only=True,
    #                      save_weights_only=False,
    #                      mode='min')

    # DEBUG:
    sv = ModelCheckpoint(CONFIG.MODEL_DIR + CONFIG.MODEL + '-{epoch:03d}-{val_loss:.4f}.h5',
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=False,
                        save_weights_only=False,
                        mode='min')

    sv_best = ModelCheckpoint(CONFIG.MODEL_DIR + CONFIG.MODEL + '-optimal.h5',
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='min')

    csv = tf.keras.callbacks.CSVLogger(
        '../history/'+CONFIG.MODEL+CONFIG.TASK_ID+'.csv')

    tb = tf.keras.callbacks.TensorBoard(CONFIG.LOG_DIR, histogram_freq=1)
    # ,save_freq='epoch'
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.4,
                                                    verbose=1,
                                                    patience=2,
                                                    min_lr=1e-8)

    # ------------------------------- TRAINING LOOP ------------------------------ #

    STEP_SIZE_TRAIN = train_augmented.n//train_augmented.batch_size
    STEP_SIZE_VALID = val_augmented.n//val_augmented.batch_size

    # TODO: Create Custom Training Loop (https://www.tensorflow.org/tensorboard/get_started) for use with tf.summary()

    # NOTE: LOG GPU stats with tensorbpard too?

    history = model.fit(train_augmented,
                        epochs=CONFIG.EPOCH,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        callbacks=[es, tb, sv, sv_best, reduce_lr, csv],
                        verbose=2,
                        class_weight=class_weights,
                        validation_data=val_augmented,
                        validation_steps=STEP_SIZE_VALID)

    # ---------------------------------------------------------------------------- #
    #                                   INFERENCE                                  #
    # ---------------------------------------------------------------------------- #
    # DEBUG:
    # -------------------------- READ SAMPLE SUBMISSION -------------------------- #

    sample_submission = pd.read_csv(PATH+'sample_submission.csv')

    test_gen = ImageDataAugmentor(augment=augments("test"),preprocess_input=CLAHE)
    test_generator = test_gen.flow_from_dataframe(dataframe=sample_submission,
                                                directory=TEST_DIR,
                                                x_col='filename',
                                                y_col=None,
                                                target_size=(
                                                    CONFIG.IM_WIDTH, CONFIG.IM_HEIGHT),
                                                color_mode='rgb',
                                                class_mode=None,
                                                batch_size=1,
                                                shuffle=False,)

    STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

    MODEL_PATH = CONFIG.MODEL_DIR + CONFIG.MODEL + '-optimal.h5'

    # ---------- LOAD OPTIMAL MODEL FROM LAST RUN AND PERFORM PREDICTION --------- #

    model = tf.keras.models.load_model(MODEL_PATH)

    test_generator.reset()
    results = model.predict(test_generator, verbose=2, steps=STEP_SIZE_TEST)

    # ------------------------- WRITE PREDICTIONS TO FILE ------------------------ #

    predicted_class_indices = np.argmax(results, axis=1)

    labels = (train_augmented.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    filenames = test_generator.filenames
    submission = pd.DataFrame({"Filename": [filename.replace('all_classes/', '')for filename in filenames],
                            "cultivar": predictions})

    v_loss = history.history['val_loss']

    submission_name = CONFIG.SAVE_DIR+'submission-'+CONFIG.MODEL + \
        '-'+str(CONFIG.WEIGHTS)+'-'+CONFIG.TASK_ID+'.csv'
    submission.to_csv(submission_name, index=False)

    # ----------------------------- SUBMIT TO KAGGLE ----------------------------- #
    msg = CONFIG.MODEL+'-'+str(CONFIG.BATCH_SIZE)+'-'+CONFIG.TASK_ID

    os.system('kaggle competitions submit -c sorghum-id-fgvc-9 -f ' +
            submission_name+' -m "'+msg+'"')
    #!kaggle competitions submit -c sorghum-id-fgvc-9 -f $submission_name -m "With flow from dataframe generator"
