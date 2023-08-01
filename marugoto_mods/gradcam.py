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
import sys

if (p := "./RetCCL") not in sys.path:
    sys.path = [p] + sys.path
from RetCCL import ResNet
import torch.nn as nn
import gdown

def load_model(model:str):
    """
    this function takes in a string describing the model and then loads said model. If model is a path to a marugoto export.pkl, it loads that. If model is 'RetCCL' it loads that and downloads the weights.
    inputs:
        model - 'RetCCL' or path to export.pkl
    returns:
        base_model - the loaded model in eval mode
    """
    # if model is RetCCL, load it and weights
    if model == 'RetCCL':
        base_model = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
        url = 'https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL'
        destination = './RetCCL'
        gdown.download_folder(url=url,output=destination,quiet=True)
        pretext_model = torch.load("./RetCCL/best_ckpt.pth")#, map_location=device)
        base_model.fc = nn.Identity()
        base_model.load_state_dict(pretext_model, strict=True)
        # group the layers for ease
        base_model.prep = nn.Sequential(base_model.conv1,base_model.bn1,base_model.relu,base_model.maxpool)
        base_model.layers = nn.Sequential(base_model.layer1,base_model.layer2,base_model.layer3,base_model.layer4)
        base_model.head = nn.Sequential(base_model.avgpool,base_model.flatten,base_model.fc)
    # if model is a pth to an export.pkl, load the fastai learner stored there and get the model out
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

    # populate map with activation values based on coords TODO: allow different magnifications/tile dims
    for i in range(len(coords)):
        x_start = coords[i,1]
        x_end = x_start + int(256/(16*float(slide.properties[openslide.PROPERTY_NAME_MPP_X])))
        y_start = coords[i,0]
        y_end = y_start + int(256/(16*float(slide.properties[openslide.PROPERTY_NAME_MPP_X])))
        map[x_start:x_end,y_start:y_end] = activations[i]
    return map

def get_tile(slide_path:Path,
             coords:tuple):
    """
    function to open the WSI and get the tile from the specified coords. TODO: generalise to different size tiles
    inputs:
        slide_path - path to WSI being analysed
        coords - coordinates of the desired tile
    returns:
        tile - the tile at the specified coordinates in RGB format and resized to fit ResNet
    """
    slide = openslide.OpenSlide(slide_path)
    tile = slide.read_region(coords,0,(256,256))
    print('Tile loaded')
    return tile.convert('RGB').resize((224,224))

def gcam_tile(tile,
              MIL_model,
              extractor):
    """
    function to load a tile, put it through feature extraction & marugoto, then back propagate to find important parts on the tile
    inputs:
        tile - tile to be analysed
        MIL_model - MIL model used in earlier steps
        extractor - feature extractor used to generate feats TODO: allow other extractors?
    returns:
        activations - activations from the last convolutional layer of the feature extractor TODO: generalise to other layers?
    """
    # run on gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make sure all models on the same device
    MIL_model.to(device)
    extractor.to(device)

    # pre-process tile
    Tforms = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tile = Tforms(tile).unsqueeze(0).to(device)
    tile.requires_grad = True

    # put tile through ResNet
    x = extractor.prep(tile)
    x = extractor.layers(x)
    x.retain_grad()
    z = extractor.head(x)

    # put features through MIL_model
    z = MIL_model.encoder(z)
    attention = MIL_model._masked_attention_scores(z,torch.tensor(z.shape[1],device=device))
    z = (attention*z).sum(-2)
    y = MIL_model.head(z)

    # do back propagation
    y[0,1].backward()
    activations = nn.functional.relu(x*x.grad).squeeze().sum(0).detach().cpu()
    return activations

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
    outputs:
        slide-level-gradcam.jpg - activation map at the slide level
        tile-level-gradcam.jpg - activation maps at the tile level for top 5 tiles TODO: generalise to n tiles?
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
    attention = MIL_model._masked_attention_scores(x,torch.tensor(x.shape[1],device=device))
    x2 = (attention*x)
    x2.retain_grad()
    y = MIL_model.head(x2.sum(-2))

    # do the backward propagation
    y[0,1].backward()
    activations = nn.functional.relu(x*x.grad).squeeze().sum(-1).detach().cpu()
    act2 = nn.functional.relu(x2*x2.grad).squeeze().sum(-1).detach().cpu()
    map = reshape_activation_map(np.array(activations),coords,slide_path,outpath)
    att_map = reshape_activation_map(np.array(act2),coords,slide_path,outpath)

    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Feats')
    plt.imshow(map,cmap='plasma')
    plt.subplot(1,2,2)
    plt.title('Attention-weighted')
    plt.imshow(att_map,cmap='plasma')
    plt.savefig(os.path.join(outpath,'slide-level-gradcam.jpg'))
    print('Slide-level GradCAM complete, starting tile-level...')

    # get top tiles
    n_tiles = 5
    plt.figure()
    idxs = np.argsort(np.array(activations))[::-1][:n_tiles]
    j = 1
    # load RetCCL Resnet model
    extractor = load_model('RetCCL').to(device)
    for idx in idxs:
        tile_coords = coords[idx]
        tile = get_tile(slide_path,tile_coords)

        # run the tile through the next step
        activations = gcam_tile(tile,MIL_model,extractor)
        
        plt.subplot(2,n_tiles,j)
        plt.title('Tile')
        plt.imshow(tile)
        plt.subplot(2,n_tiles,j+n_tiles)
        plt.title('Activations')
        plt.imshow(activations,cmap='plasma')
        j +=1
    plt.savefig(os.path.join(outpath,'tile-level-gradcam.jpg'))
    print('Complete')
