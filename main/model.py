import torch
import torch.nn as nn

class ConvolutionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"  # Default device is CPU

        self.convlayers = nn.Sequential(
            nn.Conv2d(3, 16, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)      
        )
        self.linearlayers = nn.Sequential(
            nn.Linear(119072, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = self.convlayers(x)
        x = torch.flatten(x, 1)
        return self.linearlayers(x)
