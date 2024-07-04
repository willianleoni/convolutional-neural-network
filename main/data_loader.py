import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision import transforms
import json


# Loadging JSON config
def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

config_path = "/home/admpdi/Documentos/GitHub/cnnwheeltype/main/config.json"
config = load_config(config_path)

def get_train_loader(train_dataset_path, batch_size, mean, std, data_augmentation=config['training']['data_augmentation']): ## Load the train data and aplly the transforms.
    if data_augmentation:
        transform = transforms.Compose([
            transforms.Resize(254), # Size of each image.
            transforms.RandomRotation(degrees=30), # Aplly a random chance of rotating a image in 30 degrees.
            transforms.RandomHorizontalFlip(p=0.5), # Randon chance of flip the image.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Randon chance of increase the brightnes/contrast/saturation of the image in 0.20.
            transforms.ToTensor(), # Transforms to tensor.
            transforms.Normalize(mean, std) # Normalizes the image.
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(254),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    train_data = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=transform) # Call the path and aply the transforms.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True) # Load the data in batches.

    return train_loader

def get_test_loader(test_dataset_path, batch_size, mean, std): ## Load the test data and aplly the transforms.
    transform = transforms.Compose([
         transforms.Resize(254),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)
    ])
    
    test_data = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=transform) # Call the path and aply the transforms.
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size) # Load the data in batches.
    return test_loader
