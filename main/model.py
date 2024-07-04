import torch
import torch.nn as nn
import torch.nn.functional as F
from device import set_device

device = set_device()

class ConvolutionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = device

        self.convlayers = nn.Sequential(
            nn.Conv2d(3, 16, 4), # Input channels: 3 (RGB), Output channels: 16, Kernel size: 4x4
            nn.ReLU(), # Activation function
            nn.MaxPool2d(2, 2), # Max pooling with kernel size: 2x2 and stride: 2
            nn.Conv2d(16, 32, 4), # Input channels: 16, Output channels: 32, Kernel size: 4x4
            nn.ReLU(), # Activation function
            nn.MaxPool2d(2, 2), # Max pooling with kernel size: 2x2 and stride: 2      
        )
        self.linearlayers = nn.Sequential(
            nn.Linear(119072, 128), # Fully connected layer with input size and output size
            nn.ReLU(), # Activation function
            nn.Linear(128, 3),  # Fully connected layer with input sizeand output size
        ) 

    def forward(self, x):
        x = self.convlayers(x)  # Pass the input tensor through the convolutional layers
        x = torch.flatten(x, 1) # Flatten the output tensor to prepare for the fully connected layers
        return self.linearlayers(x) # Pass the flattened tensor through the linear layers to get the final output