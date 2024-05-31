# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:42:39 2024

@author: afisher
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 14:00:10 2023

@author: afisher
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from Utils.pytorch import DigitsDataset, CNN


    
    

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper parameters
num_epochs = 40
batch_size = 5
learning_rate = 0.002


# Read dataset into train and test sets
dataset = DigitsDataset('Datasets/expanded_digits.csv')

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


# Create loaders
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)


# Train model
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, sample in enumerate(train_loader):
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
        if (i+1) % 100 == 0:
            print( f'Epoch {epoch+1}/{num_epochs}, Step {i+1}/{n_total_steps}, Loss: {loss.item():.4f}')
            
print('Finished training')

# %% Validate
classes = list(range(10))
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(10)]
    n_class_samples = [0 for _ in range(10)]
    
    for sample in test_loader:
        pixels = sample['pixels'].to(device)
        labels = sample['labels'].to(device)
        outputs = model(pixels)

        _, predicted = torch.max(outputs, 1)
        n_samples += len(labels)
        n_correct += (predicted==labels).sum().item()
        
        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if (label==pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
            
    
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of network: {acc:.2f} %')
    
    for i, class_ in enumerate(classes):
        acc = 100 * n_class_correct[i]/n_class_samples[i]
        print(f'Accuracy of {class_}: {acc:.2f}%')
        
# %% Save model
torch.save(model, 'Models/cnn.pkl')