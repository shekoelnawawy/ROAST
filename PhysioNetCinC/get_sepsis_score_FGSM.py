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
import joblib
import sys

sys.path.append('../URET')
from URET.uret.utils.config import process_config_file
cf = "URET/brute.yml"


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
    backcast_length = data.shape[0]
    nv = data.shape[1]
    
    norm = np.array([2.800e+02, 1.000e+02, 5.000e+01, 3.000e+02, 3.000e+02, 3.000e+02, 1.000e+02,
                     1.000e+02, 1.000e+02, 5.500e+01, 4.000e+03, 7.930e+00, 1.000e+02, 1.000e+02,
                     9.961e+03, 2.680e+02, 3.833e+03, 2.790e+01, 1.450e+02, 4.660e+01, 3.750e+01,
                     9.880e+02, 3.100e+01, 9.800e+00, 1.880e+01, 2.750e+01, 4.960e+01, 4.400e+02,
                     7.170e+01, 3.200e+01, 2.500e+02, 4.400e+02, 1.760e+03, 2.322e+03, 1.000e+02,
                     1.000e+00, 1.000e+00, 1.000e+00, 2.399e+01, 3.360e+02])
                     
    # Normalize data early since model expects normalized features
    data_normalized = data / norm

    # Call URET here
    if adversary:
        # --- FGSM Attack ---
        eps = 0.01  # Perturbation magnitude (applies to the normalized space)

        # Convert to tensor and enable gradient tracking
        data_tensor = torch.tensor(data_normalized, dtype=torch.float32, device=device, requires_grad=True)

        # PyTorch cuDNN requires train mode for RNN backward passes
        model.train()

        # Forward pass on normalized data
        logits, _ = model(data_tensor)

        # Target label: 0 (healthy) — attack pushes predictions away from sepsis
        target = torch.tensor([0], dtype=torch.long).to(device)
        loss = F.cross_entropy(logits, target)

        # Backward pass to compute gradients w.r.t. input
        model.zero_grad()
        loss.backward()

        # Revert back to evaluation mode
        model.eval()

        # FGSM perturbation: targeted attack minimizes loss toward class 0 
        with torch.no_grad():
            data_adv = data_tensor - eps * data_tensor.grad.sign()
            # Removing the 1.0 clamp since values in normalized space might slightly exceed 1.0 normally.
            data_adv = torch.clamp(data_adv, min=0.0)

        data_normalized = data_adv.cpu().numpy().reshape(backcast_length, nv)
        
        # Save UNNORMALIZED adversarial data back out so it matches expected format
        adversarial_data[backcast_length - 1] = (data_normalized * norm)[backcast_length - 1]

    # Nawawy's start
    data_tensor_final = torch.Tensor(data_normalized).float().to(device)
    # Nawawy's end
    threshold = 0.10
    _, probs = model(data_tensor_final)
    probs = probs[:, 1]
    argmax = (probs > threshold)
    # Nawawy's start
    return probs.cpu().data.numpy()[0], argmax.cpu().data.numpy()[0]
    # Nawawy's end

def load_sepsis_model():
    modelpth = './model_1561740354_cv_0_16_0.09277561709115473'
    # Nawawy's start
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Model(40,100,2, device)
    # Nawawy's end
    model.load_state_dict(torch.load(modelpth))
    model.eval()
    return model
