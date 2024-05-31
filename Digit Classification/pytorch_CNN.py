# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 14:00:10 2023

@author: afisher
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from Utils.pytorch import DigitsDataset, CNN
from sklearn.model_selection import StratifiedKFold
import numpy as np

def compute_accuracy(device, model, train_loader):
    n_correct = 0
    n_samples = 0
    
    for sample in train_loader:
        pixels = sample['pixels'].to(device)
        labels = sample['labels'].to(device)
        outputs = model(pixels)

        _, predicted = torch.max(outputs, 1)
        n_samples += len(labels)
        n_correct += (predicted==labels).sum().item()
        
    return 100.0 * n_correct / n_samples
    
def train_model(device, data_loader):
    # Train model
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=10e-5)
    
    n_total_steps = len(data_loader)
    for epoch in range(num_epochs):
        for i, sample in enumerate(data_loader):
            pixels = sample['pixels'].to(device)
            labels = sample['labels'].to(device)
            
            # Forward pass
            outputs = model(pixels)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Printout
            # if (i+1) % 10 == 0:
                # print( f'Epoch {epoch+1}/{num_epochs}, Step {i+1}/{n_total_steps}, Loss: {loss.item():.4f}')
    return model
    

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper parameters
num_epochs = 10
batch_size = 15
learning_rate = 0.005


# Read dataset into train and test sets
dataset = DigitsDataset('Datasets/expanded_digits.csv')
# %%
fold_train_scores = []
fold_test_scores = []
skf = StratifiedKFold(n_splits = 5)
for fold, (train_mask, test_mask) in enumerate(skf.split(dataset.data, dataset.labels)):
    print(f"Fold {fold + 1}")
    
    
    train_dataset = Subset(dataset, train_mask)
    test_dataset  = Subset(dataset, test_mask)
    
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
    
    model = train_model(device, train_loader)

                
    
    # %% Validate
    classes = list(range(10))
    with torch.no_grad():
        
        train_accuracy = compute_accuracy(device, model, train_loader)
        fold_train_scores.append(train_accuracy)
        print(f'Train: {train_accuracy:.2f} %')
        
        
        test_accuracy = compute_accuracy(device,model,  test_loader)
        fold_test_scores.append(test_accuracy)
        print(f'Test: {test_accuracy:.2f} %')
        

# %%
training_scores = np.array(fold_train_scores)
test_scores = np.array(fold_test_scores)

print(f'Training: {training_scores.mean():.2f} +/- ({training_scores.std():.2f}) %')
print(f'Testing: {test_scores.mean():.2f} +/- ({test_scores.std():.2f}) %')

# %% Final training and save
data_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
model = train_model(device, data_loader)

torch.save(model, 'Models/cnn.pkl')