import torch

def set_device():
    if torch.cuda.is_available():
                device = "cuda"
                print("Using CUDA (GPU) for computations.")
    else:
                device = "cpu"
                print("CUDA (GPU) is not available. Using CPU for computations.")

    return device

# Its important to know if you're working on GPU or CPU, this function will print you "device as ~cuda~"" if it's disponible, else, will print "device as ~cpu~"
# Pytorchs always works on the GPU when cudas available, but this function let you know when something goes wrong with your config for some reason.