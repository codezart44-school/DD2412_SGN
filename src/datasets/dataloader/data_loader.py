import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import torchvision.transforms as transforms

class DataLoaderCIFAR10:

    # Normalization parameters
    mean = [0.4914, 0.4822, 0.4465] # RGB means
    std = [0.2470, 0.2435, 0.2616]  # RGB stds

    # Define transformations including normalization
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]) # can add transforms for data augmentation
    transform_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]) # NO AUGMENTATION ALLOWED

    @classmethod
    def get_loader(cls, transform_train=None, transform_test=None, data_dir='data/', batch_size=128):

        if transform_train is None:
            transform_train = cls.transform_train
        
        if transform_test is None:
            transform_train = cls.transform_test
            
        # Create PyTorch datasets
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)

        # Create PyTorch dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, test_loader

    
if __name__ == '__main__':

    train_loader, test_loader = DataLoaderCIFAR10.get_loader()

    # Iterate over DataLoader to verify
    for images, labels in train_loader:
        print(f"Batch image shape: {images.shape}, dtype: {images.dtype}")  # Should be (64, 3, 32, 32), float32
        print(f"Batch label shape: {labels.shape}, dtype: {labels.dtype}")  # Should be (64,), int64
        break


# SCRAP
#transforms.RandomCrop(32, padding=4), # Data augmentation: Random cropping 
#transforms.RandomHorizontalFlip(), # Data augmentation: Horizontal flipping