# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:21:14 2024

@author: afisher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset


class DigitsDataset(Dataset):
    '''Digits dataset, inherit from pytorch Dataset class.'''

    def __init__(self, csv_file, transform=None):

        self.digit_pixels = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.digit_pixels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # First column is label digit
        label = int(self.digit_pixels.iloc[idx, 0])
        features = self.digit_pixels.iloc[idx, 1:].values.astype(float).reshape(1,16,16)
        
        sample = {'pixels': torch.tensor(features, dtype=torch.float32),
                  'labels': torch.tensor(label, dtype=torch.long)}
        
        
        if self.transform:
            sample = self.transform(sample)

        return sample


class CNN(nn.Module):
    '''Create convolutional neural-network'''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 16, 3,  padding=1)
        self.pool = nn.MaxPool2d(2,2) #Reduces H,W by factor of 2
        
        self.fc1 = nn.Linear(16*4*4, 40)
        self.fc2 = nn.Linear(40, 10)
        
    
    def forward(self, x):
        '''
        Apply conv/relu/pool operation twice. Then map vector to 10 outputs
        with couple fully connected layers.
        '''
        
        # 1, 16, 16
        x = self.pool(F.relu(self.conv1(x)))
        
        # 4, 8, 8
        x = self.pool(F.relu(self.conv2(x)))
        
        # 16, 4, 4
        x = x.view(-1, 16*4*4) # collapse into vector
        
        # Fully connected layers down to final 10 outputs
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x