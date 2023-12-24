# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 13:11:15 2023

@author: afisher
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Uses gpu with cuda if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Model hyperparameters
input_size = 784 # 28x28 images
hidden_size = 100 # variable
num_classes = 10 # 0-9 digits
num_epochs = 2
batch_size = 100
learning_rate = 0.001


# Import NMIST (download if necessary)
train_dataset = torchvision.datasets.MNIST(root = './data', train=True, 
   download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root = './data', train=False, 
   transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = batch_size,
   shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = batch_size,
   shuffle=True)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        
        self.l1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.act1(out)
        out = self.l2(out)
        # Softmax already included in cross_entropy loss
    
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device) 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# Training Loop
num_i_steps = len(train_loader)
for epoch in range(num_epochs):
    
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images from (100, 1, 28, 28) to (100, 784)
        images = images.reshape(-1, 784).to(device) #push to gpu if available
        labels = labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss
        if (i+1)%100 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{num_i_steps}, loss = {loss.item():.4f}')
        


# Testing/Evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    
    for i, (images, labels) in enumerate(test_loader):
        # Reshape images from (100, 1, 28, 28) to (100, 784)
        images = images.reshape(-1, 784).to(device) #push to gpu if available
        labels = labels.to(device)    
    
        # Make predictions
        outputs = model(images) #trained model
        _, predictions = torch.max(outputs, 1) #returns max, argmax
        n_samples += labels.shape[0]
        n_correct += (predictions==labels).sum().item()
        
    acc = 100.0 * (n_correct/n_samples)
    print(f'Accuracy = {acc:.2f}')



