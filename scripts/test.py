import datetime
import pprint
import numpy as np
devices = [0,1]

class CONFIG(object):
    TASK = "sorghum_id"
    TASK_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    MODEL = "EfficientNetB4"
    WEIGHTS = "imagenet"  # WEIGHTS = None
    OPTIMISER = "adam"

    IM_HEIGHT = 512
    IM_WIDTH = 512
    IM_CHANNELS = 3

    NUM_CLASSES = 100
    NUM_EPOCHS = 30
    KFOLDS = 5
    IMAGES_PER_GPU = 8
    NUM_GPUS = len(devices)
    BATCH_SIZE = IMAGES_PER_GPU*NUM_GPUS

    STEPS_BEFORE_LR_DECAY = 2
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_DELTA = 0.01

    LR = 1e-3
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

if __name__=="__main__":

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
    SorghumCONFIG = CONFIG()


    if args.path is not None:
        PATH = args.path
        print("Path has been set (value is %s)" % args.path)

    if args.batch is not None:
        SorghumCONFIG.IMAGES_PER_GPU = args.batch
        print("Images per GPU has been set (value is %s)" % args.batch)

    if args.model is not None:
        SorghumCONFIG.MODEL = args.model
        print("Model has been set (value is %s)" % args.model)

    if args.opt is not None:
        SorghumCONFIG.OPTIMISER = args.opt
        print("Optimiser has been set (value is %s)" % args.opt)

    if args.size is not None:
        SorghumCONFIG.IM_HEIGHT = args.size
        SorghumCONFIG.IM_WIDTH = args.size
        print("Image Size has been set (value is %s)" % args.size)

    if args.epochs is not None:
        SorghumCONFIG.NUM_EPOCHS = args.epochs
        print("Num. Epochs has been set (value is %s)" % args.epochs)

    if args.lr is not None:
        SorghumCONFIG.LR = args.lr
        print("Learning Rate has been set (value is %s)" % args.lr)

    if args.lr_min is not None:
        SorghumCONFIG.LR_MIN = args.lr_min
        print("Minimum Learning Rate has been set (value is %s)" % args.lr_min)

    if args.lr_steps is not None:
        SorghumCONFIG.STEPS_BEFORE_LR_DECAY = args.lr_steps
        print("Learning rate Decay Patience has been set (value is %s)" % args.lr_steps)

    if args.es_steps is not None:
        SorghumCONFIG.EARLY_STOPPING_PATIENCE = args.es_steps
        print("Early Stopping Patience has been set (value is %s)" % args.es_steps)

    if args.es_delta is not None:
        SorghumCONFIG.EARLY_STOPPING_DELTA = args.es_delta
        print("Early Stopping Delta has been set (value is %s)" % args.es_delta)
        
    SorghumCONFIG.display()