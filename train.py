import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/multimodal/FreQNet.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    # model = YOLO('runs/OGSOD/train/512/FCAIQNet/weights/last.pt') ## Resume traing setting
    model.train(data='dataset/DroneVehicle.yaml',
                cache=False,
                imgsz=640,
                # imgsz=512,
                epochs=200,
                batch=8,
                close_mosaic=0,
                workers=4, 
                # device='0,1',
                # optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # Resume setting
                amp=False,
                # fraction=0.2,
                project='runs/DroneVehicle/train',
                name='FreQNet',
                # val = False,
                )
