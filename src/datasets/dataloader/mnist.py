import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.dataloader
from torchvision import transforms
import torchvision


#================================================= Fashion MNIST =============================================#

class DataLoaderFashionMNIST:
    label_dict = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # Transformations
    transform_train = transforms.Compose([transforms.ToTensor(),]) # Only one layer (no RGB), already standardized values
    transform_test  = transforms.Compose([transforms.ToTensor(),]) # NOTE! NO AUGMENTATION ALLOWED

    @classmethod
    def get_loaders(
        cls, 
        root='./data/', 
        transform_train=transform_train, 
        batch_size=128
        ) -> tuple[torch.utils.data.DataLoader]:
        """
        A getter function that returns both the train and test `torch.utils.data.DataLoader` objects in a tuple. 

        ...

        Args:
            root (str): The path to the directory where the FashionMNIST data is downloaded.
            transform_train (torchvision.transform.Compose): A list of transforms to apply in sequence to the dataset data.
            batch_size (int): The number of samples included in each batch. 

        Returns:
            tuple[torch.utils.data.DataLoader]: A tuple of `DataLoader` objects for the train and test datasets. (train, test)
        """
        assert isinstance(transform_train, transforms.Compose)
        # Datasets
        dataset_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform_train)
        dataset_test  = torchvision.datasets.FashionMNIST(root=root, train=False, download=False, transform=cls.transform_test)
        # Data loaders
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, shuffle=True, batch_size=batch_size, num_workers=0)
        dataloader_test  = torch.utils.data.DataLoader(dataset=dataset_test, shuffle=False, batch_size=batch_size, num_workers=0)

        return (dataloader_train, dataloader_test)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataloader_train, dataloader_test = DataLoaderFashionMNIST.get_loaders()

    for batch in dataloader_train:
        images, labels = batch
        # plot first image sample
        plt.imshow(images[0].squeeze())
        plt.title(f'Label: {DataLoaderFashionMNIST.label_dict[labels[0]]}')
        plt.show()
        quit()

        



# SCRAP
#transforms.RandomCrop(32, padding=4), # Data augmentation: Random cropping 
#transforms.RandomHorizontalFlip(), # Data augmentation: Horizontal flipping