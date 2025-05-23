import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.dataloader
from torchvision import transforms
import torchvision

#================================================= CIFAR10/-100 =============================================#

class DataLoaderCIFAR10:
    # Normalization parameters
    mean = [0.4914, 0.4822, 0.4465] # RGB means
    std = [0.2470, 0.2435, 0.2616]  # RGB stds
    # Define transformations including normalization
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]) # can add transforms for data augmentation
    transform_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]) # NOTE! NO AUGMENTATION ALLOWED

    @classmethod
    def get_loaders(
        cls, 
        root='./data/', 
        transform_train=transform_train, 
        batch_size=128,
        ) -> tuple[torch.utils.data.DataLoader]:
        """
        A getter function that returns both the train and test `torch.utils.data.DataLoader` objects at the same time.

        ...

        Args:
            root (str): Path to directory where CIFAR-10 has been downloaded
            trainsform_train (torchvision.transforms.Compose): List of transforms to apply to train dataset. (E.g. to tensor, normalization or augmentation.)
            batch_size (int): Number of samples in each dataset batch. 

        Returns:
            tuple[torch.utils.data.DataLoader]: A tuple of the train and test dataloader object (train_loader, test_loader). 
        """
        assert isinstance(transform_train, transforms.Compose)
        # Create PyTorch datasets
        train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=cls.transform_test)
        # Create PyTorch dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return (train_loader, test_loader)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataloader_train, dataloader_test = DataLoaderCIFAR10.get_loaders()

    for batch in dataloader_train:
        images, labels = batch
        # plot first image sample
        plt.imshow(images[0].squeeze())
        plt.title(f'Label: {DataLoaderCIFAR10.label_dict[labels[0]]}')
        plt.show()
        quit()
