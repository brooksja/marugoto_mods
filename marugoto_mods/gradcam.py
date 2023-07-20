# Functions to enable generating heatmaps of attention & score for WSIs using the gradcam method

import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import os
from fastai.learner import load_learner
import fastai
import pandas as pd
import torch
import numpy as np
import PIL.Image as Image
import torchvision.transforms as T
import openslide

from RetCCL.ResNet import resnet50
import torch.nn as nn
import gdown

def load_model(model:str):
    # if model is RetCCL, load it and weights
    if model == 'RetCCL':
        base_model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
        url = 'https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL'
        destination = './RetCCL'
        gdown.download_folder(url=url,output=destination,quiet=True)
        pretext_model = torch.load("./RetCCL/best_ckpt.pth")#, map_location=device)
        base_model.fc = nn.Identity()
        base_model.load_state_dict(pretext_model, strict=True)
        base_model.prep = [base_model.conv1,base_model.bn1,base_model.relu,base_model.maxpool]
        base_model.layers = [base_model.layer1,base_model.layer2,base_model.layer3,base_model.layer4]
        base_model.head = [base_model.avgpool,base_model.flatten,base_model.fc]
    elif model.endswith('.pkl'):
        learn = load_learner(model)
        base_model = learn.model
    print('model & weights loaded')
    return base_model.eval()

def get_data(slide_path: Path,
             feat_dir: Path,
            ):
    """
    this function should take in the slide path and feature dir and return the feats and coords from the correct h5 file
    inputs:
        slide_path - path to the slide
        feat_dir - path to the directory containing feature h5 files
    returns:
        feats - numpy array of features
        coords - numpy array of associated tile coordinates that feats came from
    """
    # get the slide name from the path
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    # use slide name to build the path to the appropriate feats file
    feat_file = os.path.join(feat_dir,f'{slide_name}.h5')
    # get the data from the feats file
    with h5py.File(feat_file,'r') as f:
        feats = f['feats'][:]
        coords = f['coords'][:]
    return feats, coords

def reshape_activation_map(activations: np.array,
                           coords: np.array,
                           slide_path: Path,
                           outpath: Path
                           ):
    """
    function that will reshape the gradcam output based on the coordinates of the features
    inputs:
        activations - the gradcam output as a numpy array
        coords - the coordinates obtained from get_data
        slide_path - path to the slide the feats were extracted from
        outpath - path for all outputs
    returns:
        map - the activations reshaped to match the saved thumbnail
    """

    # first we use openslide to open the slide and get the dimensions
    slide = openslide.OpenSlide(slide_path)
    WSI_dims = slide.dimensions # dimensions of the slide at level 0 (used for extraction)
    thumb_dims = slide.level_dimensions[slide.get_best_level_for_downsample(16)] # dimensions at thumbnail magnification
    # create and save thumbnail image
    thumb = slide.get_thumbnail(thumb_dims).save(os.path.join(outpath,'slide_thumb.jpg'))

    # create scale factor for converting between WSI_dims and thumb_dims
    SF = 16 # set by slide.get_best_level_for_downsample(16) above
    
    # convert coords from WSI to thumb scale
    coords = coords//SF
    
    # create transparent "slide"
    map = np.zeros(thumb_dims).transpose()
    
    # populate map with activation values based on coords
    for i in range(len(coords)):
        x_start = coords[i,1]
        x_end = x_start + 224
        y_start = coords[i,0]
        y_end = y_start + 224
        map[x_start:x_end,y_start:y_end] = activations[i]
    return map


def GCAM(model: Path,
         slide_path: Path,
         feat_dir: Path,
         outpath: Path
         ):
    """
    Main function to do the gradcam.
    inputs:
        model - path to MIL model .pkl folder
        slide_path - path to the slide
        feat_dir - path to the directory containing feature h5 files
        outpath - path for saving outputs
    returns:

    """
    # run on gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the feature and coordinate data
    feats, coords = get_data(slide_path,feat_dir)

    # load the MIL model
    MIL_model = load_model(model).to(device)

    # convert feats to tensor and move to device
    feats = torch.tensor(feats,dtype=torch.float).unsqueeze(0).to(device)
    feats.requires_grad = True

    # put the feats through the model apart from the head (?)
    x = MIL_model.encoder(feats)
    x.retain_grad()
    attention = MIL_model._masked_attention_scores(x,torch.tensor(len(x),device=device))
    y = (attention*x).sum(-2)
    y = MIL_model.head(y)

    # do the backward propagation
    y[0,0].backward()
    activations = (x*x.grad).squeeze().abs().sum(-1).detach().cpu()
    
    map = reshape_activation_map(np.array(activations),coords,slide_path,outpath)

    plt.imshow(map,cmap='Reds')
    plt.show()

# EOD 20/07/23: code works, no errors but output map is basically all 0's. Try with a model & WSI that produces a sensible heatmap?

GCAM('/home/james/Documents/Hoshida_prediction/results/train/HOSHIDA=S3/Run_0/export.pkl',
     '/mnt/JD/LIVER/LLOVET-LIVER-HCC/Resections/imgs/B239.mrxs',
     '/mnt/JD/LIVER/LLOVET-LIVER-HCC/Resections/clean_aug_feats/',
     '/home/james/Documents/marugoto_mods/test_results'
     )