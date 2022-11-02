import torch.cuda as cuda

if __name__ == "__main__":
    print(f"Torch is installed with cuda: {cuda.is_available()}")
