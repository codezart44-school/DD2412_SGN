from datasets.dataloader.mnist import DataLoaderCIFAR10
import torch
import torch.nn.functional as F
import numpy as np

def label_smoothing(labels, num_classes, smoothing=0.1):
    """
    Apply label smoothing to a batch of labels.

    Args:
        labels (torch.Tensor): Tensor of shape (batch_size,) containing the class indices.
        num_classes (int): Number of classes.
        smoothing (float): Smoothing factor (alpha).

    Returns:
        torch.Tensor: Smoothed label distribution of shape (batch_size, num_classes).
    """
    assert 0 <= smoothing < 1, "Smoothing value should be in [0, 1)."

    # Initialize a one-hot encoding of the labels
    one_hot = F.one_hot(labels, num_classes).float()

    # Apply label smoothing
    smooth_labels = one_hot * (1 - smoothing) + smoothing / num_classes

    return smooth_labels


def ilr_transform(p, num_classes):
    """
    Isometric log-ratio (ilr) transform of clr-transformed data.

    Args:
        clr_p (torch.Tensor): clr-transformed data of shape (batch_size, num_classes).
        basis_matrix (torch.Tensor): Basis matrix of shape (num_classes, num_classes - 1).

    Returns:
        torch.Tensor: ilr-transformed data of shape (batch_size, num_classes - 1).
    """

    def clr_transform(p):
        log_p = torch.log(p)  # Apply log to each component
        mean_log_p = torch.mean(log_p, dim=1, keepdim=True)  # Compute mean of logs across classes
        return log_p - mean_log_p  # Subtract mean log from each component
    
    def helmert_basis(num_classes):
        D = num_classes
        V = np.zeros((D, D - 1))
        for i in range(1, D):
            V[i - 1, 0:i] = -1
            V[i, i - 1] = i
        V /= np.sqrt(np.sum(V ** 2, axis=0))  # Normalize columns
        return torch.tensor(V, dtype=torch.float32)

    basis_matrix = helmert_basis(num_classes)
    clr_p = clr_transform(p)

    return torch.matmul(clr_p, basis_matrix)  # Transform into the new basis


def regression_transform(
        labels: np.ndarray,
        num_classes: int,
        smoothing: float = 0.1
    ):

    p = label_smoothing(labels, num_classes, smoothing)
    return ilr_transform(p, num_classes)



if __name__ == '__main__':
    train_loader, test_loader = DataLoaderCIFAR10.get_loaders(batch_size=128)
    for _, labels in train_loader:
        ilr_labels = regression_transform(labels=labels, num_classes=10, smoothing=0.1)
        print(ilr_labels.shape, ilr_labels[0])
        break

    
