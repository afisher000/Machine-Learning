# %%
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt
import sounddevice as sd


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
train_dataset = torchaudio.datasets.YESNO(root = './data', download=True)
test_dataset = torchaudio.datasets.YESNO(root = './data', download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = batch_size,
   shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = batch_size,
   shuffle=True)
# %%


# Listen  to example
waveform, sample_rate, labels = train_dataset[50]
sd.play(waveform.numpy()[0], sample_rate)
sd.wait()


# %%
