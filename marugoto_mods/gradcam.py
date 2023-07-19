# Functions to enable generating heatmaps of attention & score for WSIs using the gradcam method

import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import os
from fastai.learner import load_learner
import pandas as pd
import torch

def GCAM():
    learn = load_learner('/home/james/Documents/Hoshida_prediction/results/train/HOSHIDA=S3/Run_0/export.pkl')

    model = learn.model

#    print(dir(learn))

GCAM()