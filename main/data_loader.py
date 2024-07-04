import torch
import torchvision
from torchvision import transforms
import json

class DataLoader:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return config

    def get_train_loader(self, train_dataset_path, batch_size, mean, std):
        data_augmentation = self.config['training']['data_augmentation']
        if data_augmentation:
            transform = transforms.Compose([
                transforms.Resize(254),
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(254),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        train_data = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return train_loader

    def get_test_loader(self, test_dataset_path, batch_size, mean, std):
        transform = transforms.Compose([
            transforms.Resize(254),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        test_data = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
        return test_loader
