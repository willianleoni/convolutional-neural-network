from device import set_device

device = set_device()

def train(model, dataloader, loss_func, optimizer):
    model.train()
    cumloss = 0.0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad() # Clear the gradients of all optimized variables before computing gradients for a new batch
        pred = model(imgs) # Make a forward pass through the model to obtain predictions
        loss = loss_func(pred, labels) # Calculate the loss between the model's predictions and the ground truth labels
        loss.backward() # Perform backpropagation through the computation graph to compute gradients
        optimizer.step() # Update the model's parameters based on the computed gradients and the optimizer

        cumloss += loss.item() 

    return cumloss / len(dataloader)