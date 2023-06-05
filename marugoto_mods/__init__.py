from datetime import datetime
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union, Dict, Any, Tuple, TypeVar
from warnings import warn
from dataclasses import dataclass
import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from fastai.vision.learner import load_learner
import torch
from torch import nn
import torch.nn.functional as F
import h5py
from torch.utils.data import Dataset
from fastai.vision.all import (
    Learner,
    DataLoader,
    DataLoaders,
    RocAuc,
    SaveModelCallback,
    CSVLogger,
)

from marugoto.mil.data import SKLearnEncoder
from marugoto.mil._mil import train, deploy
from marugoto.mil.data import get_cohort_df, get_target_enc, make_dataset
from marugoto.data import SKLearnEncoder
from marugoto.mil.model import MILModel
