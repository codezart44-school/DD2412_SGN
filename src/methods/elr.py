import os
import torch
import torch.optim as optim
import torch.nn.functional as F  # provides functions for activation, loss etc.
from tqdm import tqdm

from ..models.wide_resnet_28_2 import create_model
from ..utils.config import *
from ..datasets.dataloader.cifar import DataLoaderCIFAR10

# Initialize p_t for storing EMA predictions
p_t = {}

def update_p_t(p_t, indices, probs, beta=0.7):
    """
    Updates the running average predictions (p_t) for ELR.
    Args:
        p_t (dict): A dictionary mapping data indices to EMA predictions.
        indices (Tensor): Batch indices of the current examples.
        probs (Tensor): Predicted probabilities for the current batch.
        beta (float): Momentum for the EMA update (0 < beta < 1).
    Returns:
        Updated p_t dictionary.
    """
    for idx, prob in zip(indices, probs):
        idx = idx.item()
        if idx not in p_t:
            p_t[idx] = prob.clone().detach()
        else:
            p_t[idx] = beta * p_t[idx] + (1 - beta) * prob.clone().detach()
    return p_t

def train_one_epoch_elr(
      model,
      loader,
      optimizer,
      epoch,
      num_epochs,
      p_t,
      lambda_elr=3.0,
      beta=0.7,
    ):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_steps = len(loader)

    # Counter for tracking indices
    batch_start_idx = 0

    for step, (images, labels) in enumerate(tqdm(loader, desc="Training", leave=False), start=1):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        # Generate indices for the current batch
        batch_size = images.size(0)
        indices = torch.arange(batch_start_idx, batch_start_idx + batch_size).to(DEVICE)
        batch_start_idx += batch_size

        # Forward pass
        logits = model(images)
        probs = F.softmax(logits, dim=1)

        # Compute ELR loss
        ce_loss = F.cross_entropy(logits, labels)
        reg_term = 0.0
        for idx, prob in zip(indices, probs):
            idx = idx.item()
            if idx in p_t:
                reg_term += torch.sum(p_t[idx] * torch.log(prob + 1e-12))
        reg_term = -lambda_elr * reg_term / logits.size(0)
        loss = ce_loss + reg_term

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update p_t (EMA predictions)
        p_t = update_p_t(p_t, indices, probs, beta=beta)

        # Track metrics
        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return running_loss / num_steps, accuracy, p_t




@torch.no_grad()
def evaluate_elr(model, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Forward pass
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        # Compute accuracy
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        running_loss += loss.item()

    accuracy = 100. * correct / total
    return running_loss / len(loader), accuracy


def train_elr(
      batch_size: int,
      lr: float,
      num_epochs: int,
      device: torch.device,
      noise_type: str = None,
      noise_rate: float = 0.0,
      warmup_epochs=5,
      lambda_elr=3.0,
      beta=0.7,
    ):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_loader, test_loader, input_shape = DataLoaderCIFAR10.get_loaders(
        root=DATA_DIR,
        download=False,
        batch_size=batch_size,
        num_workers=2,
        noise_type=noise_type,
        noise_rate=noise_rate
    )

    # Create Wide-ResNet model
    model = create_model(depth=28, width_multiplier=2, num_classes=10, input_shape=input_shape, version=1, sgn=False).to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # Warmup and learning rate scheduling
    base_lr = lr
    start_lr = base_lr * 0.1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_epochs - warmup_epochs))

    def warmup_lr_scheduler(optimizer, epoch, warmup_epochs, start_lr, base_lr):
        lr = start_lr + (base_lr - start_lr) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Initialize p_t for EMA predictions
    p_t = {}

    result_dict = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
        'hyperparams': {
            'batch_size': batch_size,
            'lr': lr,
            'num_epochs': num_epochs,
            'noise_type': noise_type,
            'noise_rate': noise_rate,
        },
    }

    for epoch in range(num_epochs):
        # Adjust learning rate
        if epoch < warmup_epochs:
            warmup_lr_scheduler(optimizer, epoch + 1, warmup_epochs, start_lr, base_lr)
        else:
            scheduler.step()

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc, p_t = train_one_epoch_elr(
            model,
            train_loader,
            optimizer,
            epoch,
            num_epochs,
            p_t,
            lambda_elr=lambda_elr,
            beta=beta
        )
        test_loss, test_acc = evaluate_elr(model, test_loader)

        result_dict['train_loss'].append(train_loss)
        result_dict['test_loss'].append(test_loss)
        result_dict['train_acc'].append(train_acc)
        result_dict['test_acc'].append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    print("Training complete!")
    return result_dict