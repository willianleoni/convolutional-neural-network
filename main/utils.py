import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sn

class Utils:
    def __init__(self, device):
        self.device = device

    def matplotlib_imshow(self, img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def plot_losses(self, losses, save_path=None):
        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        for loss_name, loss_values in losses.items():  
            ax.plot(loss_values, label=loss_name)
        ax.legend(fontsize="16")
        ax.set_xlabel("Epoch", fontsize="16")
        ax.set_ylabel("Loss", fontsize="16")
        ax.set_title("Training and Validation Losses", fontsize="16")

        # Save the plot as an image file if save_path is provided
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def make_confusion_matrix(self, model, dataloader, n_classes):
        confusion_matrix = torch.zeros(n_classes, n_classes, dtype=torch.int64)
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                for t, p in zip(torch.as_tensor(labels, dtype=torch.int64).view(-1), 
                                torch.as_tensor(predicted, dtype=torch.int64).view(-1)):
                    confusion_matrix[t, p] += 1
        return confusion_matrix

    def evaluate_accuracy(self, model, dataloader, classes, verbose=True):
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        confusion_matrix = self.make_confusion_matrix(model, dataloader, len(classes))
        if verbose:
            total_correct = 0.0
            total_prediction = 0.0
            for i, classname in enumerate(classes):
                correct_count = confusion_matrix[i][i].item()
                class_pred = torch.sum(confusion_matrix[i]).item()

                total_correct += correct_count
                total_prediction += class_pred

                accuracy = 100 * float(correct_count) / class_pred
                print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
            print("Global accuracy is {:.1f}".format(100 * total_correct / total_prediction))
        return confusion_matrix

    def plot_confusion_matrix(self, confusion_matrix, classes, save_path=None):
        fig, ax = plt.subplots(figsize=(len(classes) + 2, len(classes) + 2))
        sn.set(font_scale=1.4)
        sn.heatmap(confusion_matrix.tolist(), 
                   annot=True, annot_kws={"size": 16}, fmt='d',
                   xticklabels=classes, yticklabels=classes, ax=ax)

        ax.set_xlabel('Predicted Label', fontsize=16)
        ax.set_ylabel('True Label', fontsize=16)
        ax.set_title('Confusion Matrix', fontsize=18)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
