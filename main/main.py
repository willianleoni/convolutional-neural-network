import torch
import torch.nn as nn
from model import ConvolutionalModel
from data_loader import get_train_loader, get_test_loader
from utils import plot_losses, evaluate_accuracy, make_confusion_matrix, plot_confusion_matrix
from train import train
from device import set_device
from validate import validate
import json
import datetime
import os


# Loadging JSON config
def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

config_path = "/home/admpdi/Documentos/GitHub/cnnwheeltype/main/config.json"
config = load_config(config_path)


def main():

    # config_path = "~/home/admpdi/Documentos/GitHub/cnnwheeltype/main/config.json" # JSON Path
    # config = load_config("config.json")  # Load configurations from JSON file

    train_dataset_path = config['training']['train_dataset_path'] # Set the path you wanna use for the models train in config.JSON
    test_dataset_path = config['training']['test_dataset_path'] # Set the path you wanna use for the models test in config.JSON, its usually 20% of your train data
    device = set_device()

    # Batch size and normalization parameters
    batch_size = config['training']['batch_size'] # Set the batch size that are used for training in config.JSON.
    mean = (0.5, 0.5, 0.5) 
    std = (0.5, 0.5, 0.5)

    # Function to call the data loaders, see ~data_loader.py~ for more informations.
    train_loader = get_train_loader(train_dataset_path, batch_size, mean, std)
    test_loader = get_test_loader(test_dataset_path, batch_size, mean, std)

    # Set the names for each label.
    categorias = ["aro", "cubo", "raio"]

    # Initialize and move the model to the GPU if available, more informations on ~device.py~.
    model = ConvolutionalModel().to(device)

    # Set the model to evaluation mode
    model.eval()

    # Define the loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['learning_rate'])

    # Training and validation loop
    epochs = config['training']['num_epochs'] # Number of epochs use for train and validation
    train_losses = [] # Generates a list for the training epochs
    test_losses = [] # Generates a list for the validations epochs
    
    
    for t in range(epochs): # Loop over the epochs

        train_loss = train(model, train_loader, loss_func, optimizer) # Training the model and obtaining the training loss for the current epoch.
        train_losses.append(train_loss) # Store the training loss in the train_losses list.

        test_loss = validate(model, test_loader, loss_func) # Training the model and obtaining the validation loss for the current epoch.
        test_losses.append(test_loss) # Store the validation loss in the test_losses list.
 
        if t % 2 == 0:  # The condition "if t % 2 == 0" verifies if the t valor is a par, if true, this will print the information of the progress.
            print(f"Epoch [{t+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {test_loss:.4f}")

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Plot the training and validation losses.
    losses = {"Train loss": train_losses, "Test loss": test_losses}
    loss_plot_filename = f"loss_plot_{current_time}.png"
    loss_plot_path = os.path.join(config['paths']['loss_plot'], loss_plot_filename)
    plot_losses(losses, loss_plot_path)

    # Save the confusion matrix as an image file.
    confusion_matrix = evaluate_accuracy(model, test_loader, categorias, verbose=True)
    confusion_matrix_filename = f"confusion_matrix_{current_time}.png"
    confusion_matrix_path = os.path.join(config['paths']['confusion_matrix'], confusion_matrix_filename)
    plot_confusion_matrix(confusion_matrix, classes=categorias, save_path=confusion_matrix_path)


    model_filename = f"model_{current_time}.pth"
    model_path = os.path.join(config['paths']['model'], model_filename)
    torch.save(model.state_dict(), model_path)

    # Save hiperparameters and confusion matrix to a JSON file
    result_filename = f"results_{current_time}.json"
    results_dir = "/home/admpdi/Documentos/GitHub/cnnwheeltype/results"
    result_path = os.path.join(results_dir, result_filename)
    result_dict = {
        "hyperparameters": {
            "num_epochs": epochs,
            "learning_rate": config['training']['learning_rate'],
            "batch_size": batch_size,
            "data_augmentation": config['training']['data_augmentation']
        },
        "confusion_matrix": confusion_matrix.tolist(),
        "train_loss": train_loss,
        "test_loss": test_loss,
    }

    with open(result_path, "w") as f:
        json.dump(result_dict, f)

if __name__ == "__main__":
    main()
