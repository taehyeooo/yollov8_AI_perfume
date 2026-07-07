### read me !! ###

## Development environment

# python - 3.8
# pandas 
#
#

import pandas as pd
import random
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import cv2
from ultralytics import YOLO
import time
import sys
sys.path.append('C:/Users/우리집/Desktop/spm') #여기에 내려받은 파일 경로!!

import camera


import filter

modeldir = './runs/detect/train4/weights/best.pt'
datadir = './data/alist3.csv'

cam =camera.cameramood(modeldir)

filter.filtering_mood(datadir, cam)

