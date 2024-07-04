import torch

class DeviceManager:
    @staticmethod
    def set_device():
        if torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA (GPU) for computations.")
        else:
            device = "cpu"
            print("CUDA (GPU) is not available. Using CPU for computations.")

        return device
