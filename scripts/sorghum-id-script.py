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

tf.get_logger().setLevel('ERROR')
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

NUM_GPUS = len(devices)
base_batch_size = 16
batch_size = base_batch_size*NUM_GPUS
epoch = 100
WIDTH = 512
HEIGHT = 512
DEBUG = False


image_df = pd.read_csv(PATH+'train_cultivar_mapping.csv')
image_df.dropna(inplace=True)

if DEBUG:
    image_df=image_df[:2000]
    epoch = 20

image_df


kfold = StratifiedKFold(n_splits=3, shuffle=True)

for train_index, valid_index in kfold.split(image_df['image'],image_df['cultivar']):
    train_images, valid_images = image_df['image'].iloc[train_index], image_df['image'].iloc[valid_index]
    train_cultivar, valid_cultivar = image_df['cultivar'].iloc[train_index], image_df['cultivar'].iloc[valid_index]

train_df= pd.DataFrame({'image':train_images, 'cultivar':train_cultivar})
val_df= pd.DataFrame({'image':valid_images, 'cultivar':valid_cultivar})
print('Number of Training Samples: ',len(train_df), '    Number of Validation Samples: ', len(val_df))


transform = Compose([
            alb.RandomResizedCrop(height=HEIGHT, width=WIDTH),
            alb.Flip(p=0.5),
            alb.RandomRotate90(p=0.5),
            alb.ShiftScaleRotate(p=0.5),
            alb.HueSaturationValue(p=0.5),
            alb.OneOf([
                alb.RandomBrightnessContrast(p=0.5),
                alb.RandomGamma(p=0.5),
            ], p=0.5),
            OneOf([
                alb.Blur(p=0.1),
                alb.GaussianBlur(p=0.1),
                alb.MotionBlur(p=0.1),
            ], p=0.1),
            OneOf([
                alb.GaussNoise(p=0.1),
                alb.ISONoise(p=0.1),
                alb.GridDropout(ratio=0.5, p=0.2),
                alb.CoarseDropout(max_holes=16, min_holes=8, max_height=16, max_width=16, min_height=8, min_width=8, p=0.2)
            ], p=0.2)
        ])

train_datagen = ImageDataAugmentor(augment=transform)
val_datagen = ImageDataAugmentor()

train_augmented = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    shuffle=True,
    directory=train_dir,
    x_col='image',
    y_col='cultivar',
    class_mode='categorical',
    target_size=(HEIGHT,WIDTH),
    batch_size=batch_size)

val_augmented = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    shuffle=True,
    directory=train_dir,
    x_col='image',
    y_col='cultivar',
    class_mode='categorical',
    target_size=(HEIGHT,WIDTH),
    batch_size=batch_size)

num_classes = len(train_augmented.class_indices)
class_id, num_images = np.unique(train_augmented.classes,return_counts=True)
max_value = max(num_images)
class_weights = {c : max_value/n for c,n in zip(class_id, num_images)}


def load_model(): # Here we choose our model : Efficientnet B4 pretrained with ImageNet dataset with an input shape of 260x260
    model = tf.keras.applications.EfficientNetB4(include_top=False,weights='imagenet',input_shape=(HEIGHT,WIDTH,3))
    return model

def set_nontrainable_layers(model): # We define trainability for the base model
    model.trainable=False
    return model

def set_trainable_layers(model): 
    model.trainable=True
    return model

def set_quartertrainable_layers(model):
    model.trainable=True
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    return model
    
def set_halftrainable_layers(model):
    model.trainable=True
    for layer in model.layers[-round(len(model.layers)*0.5):]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    return model

def add_last_layers(model): # Here we complete our model with last layers for our problem at hand
    input_layer = tf.keras.Input(shape=(HEIGHT,WIDTH,3))
#     base_model = set_quartertrainable_layers(model)
#     base_model = set_halftrainable_layers(model)
#     base_model = set_nontrainable_layers(model)
    base_model = set_trainable_layers(model)
    flatten_layer = layers.Flatten()
    global_layer = layers.GlobalAveragePooling2D()
    dense_layer = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')
    dropout_layer = layers.Dropout(0.5)
    prediction_layer = layers.Dense(num_classes, activation='softmax')
    
    model = Sequential([
        input_layer,
        base_model,
        global_layer,
        dropout_layer,
#         flatten_layer,
#         dense_layer,
        prediction_layer
    ])
    return model

def build_model(): # We assemble our model and compile it with the proper loss function and metrics for image classification
    model = load_model()
    model = add_last_layers(model)
    
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy',
                 optimizer=opt,
                 metrics=[tf.keras.metrics.CategoricalAccuracy(),
                 tf.keras.metrics.AUC(),
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(),
                 tf.keras.metrics.TruePositives(),
                 tf.keras.metrics.TrueNegatives(),
                 tf.keras.metrics.FalsePositives(),
                 tf.keras.metrics.FalseNegatives()])
    return model


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = build_model()
model.summary()


es = EarlyStopping(monitor='val_accuracy',
                   patience=7,
                   verbose=1,
                   restore_best_weights=True)

cp = ModelCheckpoint(save_dir + 'effnetB4-{epoch:04d}.ckpt',
                     monitor='val_loss',
                     verbose=1,
                     save_best_only=True,
                     save_weights_only=False,
                     mode='min' )

sv = ModelCheckpoint(model_dir + 'effnetB4-{epoch:04d}.h5',
                     monitor='val_loss',
                     verbose=1,
                     save_best_only=False,
                     save_weights_only=False,
                     mode='min' )

sv_best = ModelCheckpoint(model_dir + 'effnetB4-optimal.h5',
                     monitor='val_loss',
                     verbose=1,
                     save_best_only=True,
                     save_weights_only=False,
                     mode='min' )

csv = tf.keras.callbacks.CSVLogger('../history/'+TASK_ID+'.csv')

tb = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.4,
                                                 verbose=1,
                                                 patience=2, 
                                                 min_lr=1e-6)


STEP_SIZE_TRAIN = train_augmented.n//train_augmented.batch_size
STEP_SIZE_VALID = val_augmented.n//val_augmented.batch_size


model.fit(train_augmented,
          epochs=epoch,
          steps_per_epoch=STEP_SIZE_TRAIN,
          callbacks=[es,tb,sv,sv_best,reduce_lr,csv],
          verbose=2,
          class_weight=class_weights,
          validation_data=val_augmented,
          validation_steps=STEP_SIZE_VALID)


## INFERENCE

sample_submission = pd.read_csv(PATH+'sample_submission.csv')

test_gen= ImageDataAugmentor()
test_generator = test_gen.flow_from_dataframe(dataframe=sample_submission,
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

MODEL_PATH = model_dir+"effnetB4-optimal.h5"

model = tf.keras.models.load_model(MODEL_PATH)

test_generator.reset()
results = model.predict(test_generator,verbose=1,steps=STEP_SIZE_TEST)

predicted_class_indices=np.argmax(results,axis=1)

labels = (train_augmented.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
submission=pd.DataFrame({"Filename":[filename.replace('all_classes/','')for filename in filenames],
                      "cultivar":predictions})

submission_name = save_dir+'submission_EfficientnetB4ImageNet'+TASK_ID+'.csv'
submission.to_csv(submission_name,index=False)

#!kaggle competitions submit -c sorghum-id-fgvc-9 -f $submission_name -m "With flow from dataframe generator"