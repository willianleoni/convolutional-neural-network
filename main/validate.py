import torch
from device import set_device

device = set_device()

def validate(model, dataloader, loss_func):
    model.eval()
    cumloss = 0.0 # Variable to store the cumulative loss during validation.

    with torch.no_grad(): # Disable gradient computation during validation.
        for imgs, labels in dataloader: # Iterate over the data in the validation dataloader.
            imgs, labels = imgs.to(device), labels.to(device) # Move the images and labels to the device, see device.py

            pred = model(imgs) # Perform forward pass to obtain predictions.
            loss = loss_func(pred, labels) # Calculate the loss between predictions and ground truth labels.
            cumloss += loss.item()  # Accumulate the loss for the current batch.

    return cumloss / len(dataloader)