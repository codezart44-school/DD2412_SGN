# imports
from torchvision import datasets, transforms
import os

#=========================================================================================#

def get_transform(normalize=False, mean=None, std=None):
    """..."""
    transforms_list = [transforms.ToTensor()]
    if normalize == True:
        transforms_list.append(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(transforms_list)
    return transform

def download_mnist(root='./data'):
    """..."""
    transform = get_transform()
    mnist_train = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    return mnist_train, mnist_test

def download_cifar10(root='./data'):
    """..."""
    transform = get_transform()
    cifar10_train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    return cifar10_train, cifar10_test

def download_cifar100(root='./data'):
    """..."""
    transform = get_transform()
    cifar100_train = datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    cifar100_test = datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    return cifar100_train, cifar100_test

#=========================================================================================#

