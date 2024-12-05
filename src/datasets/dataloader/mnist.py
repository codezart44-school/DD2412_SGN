import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.dataloader
from torchvision import transforms
import torchvision


def print_matrix(matrix):
    s = [[str(round(e, 4)) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


#================================================= Fashion MNIST =============================================#

class NoisyFashionMNIST(Dataset):
    def __init__(self, dataset, noise_type=None, noise_rate=0.0, transition_matrix=None):
        """
        A wrapper around the FashionMNIST dataset to add optional noise to the labels.

        Args:
            dataset (torch.utils.data.Dataset): The original FashionMNIST dataset.
            noise_type (str, optional): Type of noise to apply. Options: 'symmetric', 'asymmetric'. Default is None.
            noise_rate (float): The proportion of labels to corrupt (between 0 and 1).
            transition_matrix (np.ndarray, optional): A custom transition probability matrix for asymmetric noise.
        """
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.num_classes = len(DataLoaderFashionMNIST.label_dict)
        self.transition_matrix = transition_matrix
        self._apply_noise()

    def _apply_noise(self):
        """
        Applies the specified noise to the dataset's labels.
        """
        if self.noise_type is None or self.noise_rate <= 0.0:
            # No noise to apply
            self.noisy_labels = self.dataset.targets.clone()  # Keep original labels
            return

        targets = self.dataset.targets.numpy()  # Convert to NumPy for easy manipulation
        n_samples = len(targets)

        if self.noise_type == 'symmetric':
            # Symmetric noise: Replace labels with random labels with uniform probability
            n_noisy = int(self.noise_rate * n_samples)
            noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
            noisy_labels = targets.copy()
            for idx in noisy_indices:
                original_label = targets[idx]
                possible_labels = [label for label in range(self.num_classes) if label != original_label]
                noisy_label = np.random.choice(possible_labels)
                noisy_labels[idx] = noisy_label
            self.noisy_labels = torch.tensor(noisy_labels, dtype=torch.long)

        elif self.noise_type == 'asymmetric':
            if self.transition_matrix is None:
                # Generate a default transition matrix if not provided
                self.transition_matrix = self._generate_default_transition_matrix()

            # Ensure the transition matrix is properly normalized
            assert self.transition_matrix.shape == (self.num_classes, self.num_classes)
            assert np.allclose(self.transition_matrix.sum(axis=1), 1), "Rows must sum to 1"

            noisy_labels = targets.copy()
            for i in range(len(targets)):
                true_label = targets[i]
                noisy_labels[i] = np.random.choice(
                    self.num_classes, p=self.transition_matrix[true_label]
                )
            self.noisy_labels = torch.tensor(noisy_labels, dtype=torch.long)

        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

    def _generate_default_transition_matrix(self):
        """
        Generates a default transition matrix for asymmetric noise.
        """
        num_classes = self.num_classes
        matrix = np.eye(num_classes) * (1 - self.noise_rate)  # Start with identity matrix for correct labels

        # Define plausible misclassification probabilities manually
        plausible_transitions = {
            3: [4],  # Dress -> Coat
            4: [3],  # Coat -> Dress
            2: [6],  # Pullover -> Shirt
            6: [2],  # Shirt -> Pullover
        }

        # Assign higher probabilities for plausible transitions
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:  # Off-diagonal
                    
                    if i in plausible_transitions:
                        transition_prob_factor = 0.7
                        if j in plausible_transitions.get(i):
                            matrix[i, j] = self.noise_rate * transition_prob_factor
                        else:
                            matrix[i, j] = self.noise_rate * (1-transition_prob_factor) / (num_classes - (1 + len(plausible_transitions.get(i))))

                    else:
                        matrix[i, j] = self.noise_rate / (num_classes - 1)

        # Normalize rows to sum to 1
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        print_matrix(matrix)
        return matrix

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]  # Get the original image
        label = self.noisy_labels[index]  # Get the noisy label
        return image, label


class DataLoaderFashionMNIST:
    label_dict = [
        "T-shirt/top", # 0
        "Trouser", # 1
        "Pullover", # 2
        "Dress", # 3
        "Coat", # 4
        "Sandal", # 5
        "Shirt", # 6
        "Sneaker", # 7
        "Bag", # 8
        "Ankle boot", # 9
    ]

    # Transformations
    transform_train = transforms.Compose([transforms.ToTensor(),]) # Only one layer (no RGB), already standardized values
    transform_test  = transforms.Compose([transforms.ToTensor(),]) # NOTE! NO AUGMENTATION ALLOWED

    @classmethod
    def get_loaders(
        cls, 
        root='./data/', 
        transform_train=transform_train, 
        batch_size=128,
        noise_type=None,
        noise_rate=0.0
        ) -> tuple[torch.utils.data.DataLoader]:
        """
        A getter function that returns both the train and test `torch.utils.data.DataLoader` objects in a tuple. 

        ...

        Args:
            root (str): The path to the directory where the FashionMNIST data is downloaded.
            transform_train (torchvision.transform.Compose): A list of transforms to apply in sequence to the dataset data.
            batch_size (int): The number of samples included in each batch. 
            noise_type (str): Type of noise to apply to train labels. Options: 'symmetric', 'asymmetric'.
            noise_rate (float): The proportion of labels to corrupt (between 0 and 1).

        Returns:
            tuple[torch.utils.data.DataLoader]: A tuple of `DataLoader` objects for the train and test datasets. (train, test)
        """
        assert isinstance(transform_train, transforms.Compose)

        # Original datasets
        dataset_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform_train)
        dataset_test  = torchvision.datasets.FashionMNIST(root=root, train=False, download=False, transform=cls.transform_test)

        if noise_type is not None and noise_rate > 0.0:
            dataset_train = NoisyFashionMNIST(dataset=dataset_train, noise_type=noise_type, noise_rate=noise_rate)

        # Data loaders
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, shuffle=True, batch_size=batch_size, num_workers=0)
        dataloader_test  = torch.utils.data.DataLoader(dataset=dataset_test, shuffle=False, batch_size=batch_size, num_workers=0)

        return (dataloader_train, dataloader_test)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataloader_train, dataloader_test = DataLoaderFashionMNIST.get_loaders(noise_type='asymmetric', noise_rate=0.99)

    for batch in dataloader_train:
        images, labels = batch
        # plot first image sample
        plt.imshow(images[0].squeeze())
        plt.title(f'Label: {DataLoaderFashionMNIST.label_dict[labels[0]]} = {labels[0]}')
        plt.show()

        
