import torch
import torch.nn as nn
import json
import datetime
import os
from model import ConvolutionalModel
from data_loader import DataLoader
from utils import Utils
from train import Trainer
from validate import Validator
from device import DeviceManager

class CNNTrainer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = DeviceManager.set_device()

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return config

    def main(self):
        # Paths from the config.json file
        train_dataset_path = self.config['training']['train_dataset_path']
        test_dataset_path = self.config['training']['test_dataset_path']

        # Batch size and normalization parameters
        batch_size = self.config['training']['batch_size']
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        # Data loaders initialization
        data_loader = DataLoader(config_path)
        train_loader = data_loader.get_train_loader(train_dataset_path, batch_size, mean, std)
        test_loader = data_loader.get_test_loader(test_dataset_path, batch_size, mean, std)

        # Model initialization
        model = ConvolutionalModel().to(self.device)

        # Loss function and optimizer initialization
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config['training']['learning_rate'])

        # Training and validation process
        trainer = Trainer(self.device)
        validator = Validator(self.device)
        utils = Utils(self.device)

        train_losses = []
        test_losses = []

        for epoch in range(self.config['training']['num_epochs']):
            train_loss = trainer.train_model(model, train_loader, loss_func, optimizer)
            train_losses.append(train_loss)

            if epoch % 2 == 0:
                print(f"Epoch [{epoch+1}/{self.config['training']['num_epochs']}], Train Loss: {train_loss:.4f}")

            test_loss = validator.validate_model(model, test_loader, loss_func)
            test_losses.append(test_loss)

        # Plot and save losses
        losses = {"Train loss": train_losses, "Test loss": test_losses}
        loss_plot_path = os.path.join(self.config['paths']['loss_plot'], f"loss_plot_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
        utils.plot_losses(losses, save_path=loss_plot_path)

        # Evaluate and save confusion matrix
        categories = ["aro", "cubo", "raio"]
        confusion_matrix = utils.evaluate_accuracy(model, test_loader, categories, verbose=True)
        confusion_matrix_path = os.path.join(self.config['paths']['confusion_matrix'], f"confusion_matrix_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
        utils.plot_confusion_matrix(confusion_matrix, classes=categories, save_path=confusion_matrix_path)

        # Save model and results
        model_path = os.path.join(self.config['paths']['model'], f"model_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth")
        torch.save(model.state_dict(), model_path)

        result_dict = {
            "hyperparameters": {
                "num_epochs": self.config['training']['num_epochs'],
                "learning_rate": self.config['training']['learning_rate'],
                "batch_size": batch_size,
                "data_augmentation": self.config['training']['data_augmentation']
            },
            "confusion_matrix": confusion_matrix.tolist(),
            "train_loss": train_loss,
            "test_loss": test_loss,
        }

        result_path = os.path.join(self.config['paths']['results'], f"results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
        with open(result_path, "w") as f:
            json.dump(result_dict, f)

if __name__ == "__main__":
    config_path = "./config.json"
    trainer = CNNTrainer(config_path)
    trainer.main()
