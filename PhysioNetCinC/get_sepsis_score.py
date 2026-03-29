#!/usr/bin/env python

import pandas as pd
import numpy as np
import glob
from torch.utils.data import Dataset
import copy
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Nawawy's start
from pathlib import Path
import joblib
import sys

from URET.uret.utils.config import process_config_file
cf = str(Path(__file__).resolve().parent/"URET"/"brute.yml")

def feature_extractor(x):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return torch.tensor(x, dtype=torch.float).to(device)

def mse(output, target):
	loss=torch.mean((output - target) ** 2)
	return loss
# Nawawy's end

class Model(nn.Module):
    # Nawawy's start
    def __init__(self,input_size,hidden_size,hidden_depth, device=None):
    # Nawawy's end
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_depth = hidden_depth
        # Nawawy's start
        self.device = device
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.hidden_depth, bidirectional=False, dropout=0, device = self.device)
        self.fc1 = nn.Linear(1*self.hidden_size, 2, device = self.device)
        # Nawawy's end
    def forward(self, x):
        h0 = torch.zeros(self.hidden_depth * 1, 1, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.hidden_depth * 1, 1, self.hidden_size).to(x.device)

        x = x.view(-1, 1, self.input_size)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.fc1(x)
        x = x.view(-1, 2)
        x = x[-1, :].view(1, 2)
        return x, F.softmax(x, dim=1)


# Nawawy's start
def get_sepsis_score(data, model, adversary=False, adversarial_data=None):
# Nawawy's end
    data = pd.DataFrame(data)
    data = data.fillna(method='ffill')
    data = data.fillna(0).values
    # Nawawy's start
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # data=torch.tensor(data).float().to(device)
    backcast_length = data.shape[0]
    nv = data.shape[1]
    # Call URET here
    if adversary:
        explorer = process_config_file(cf, model, feature_extractor=feature_extractor, input_processor_list=[])
        explorer.scoring_function = mse
        allPatients_benign = data
        explore_params = [allPatients_benign, backcast_length, nv]
        allPatients_adversarial = np.array(explorer.explore(explore_params))
        if allPatients_adversarial[0] is None:
            allPatients_adversarial = allPatients_benign
        allPatients_adversarial = allPatients_adversarial.reshape(backcast_length, nv)
        data = allPatients_adversarial
        adversarial_data[backcast_length - 1] = data[backcast_length - 1]
    # Nawawy's end
    norm = [2.800e+02, 1.000e+02, 5.000e+01, 3.000e+02, 3.000e+02, 3.000e+02, 1.000e+02,
            1.000e+02, 1.000e+02, 5.500e+01, 4.000e+03, 7.930e+00, 1.000e+02, 1.000e+02,
            9.961e+03, 2.680e+02, 3.833e+03, 2.790e+01, 1.450e+02, 4.660e+01, 3.750e+01,
            9.880e+02, 3.100e+01, 9.800e+00, 1.880e+01, 2.750e+01, 4.960e+01, 4.400e+02,
            7.170e+01, 3.200e+01, 2.500e+02, 4.400e+02, 1.760e+03, 2.322e+03, 1.000e+02,
            1.000e+00, 1.000e+00, 1.000e+00, 2.399e+01, 3.360e+02]
    data = data/norm
    # Nawawy's start
    data = torch.Tensor(data).float().to(device)
    # Nawawy's end
    threshold = 0.10
    _, probs = model(data)
    probs = probs[:, 1]
    argmax = (probs > threshold)
    # Nawawy's start
    return probs.cpu().data.numpy()[0], argmax.cpu().data.numpy()[0]
    # Nawawy's end

def load_sepsis_model():
    modelpth = str(Path(__file__).resolve().parent/'model_1561740354_cv_0_16_0.09277561709115473')
    # Nawawy's start
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Model(40,100,2, device)
    # Nawawy's end
    model.load_state_dict(torch.load(modelpth))
    model.eval()
    return model
