import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..models.wide_resnet_28_2 import create_model
from ..utils.config import *
from ..datasets.dataloader.cifar import DataLoaderCIFAR10

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(loader), accuracy

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(loader), accuracy


def train_ce(
      batch_size: int,
      lr: float,
      num_epochs: int,
      device: torch.device,
      noise_type: str = None,
      noise_rate: float = 0.0,
      warmup_epochs=5
    ):

    # Save results
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

    train_loader, test_loader, input_shape = DataLoaderCIFAR10.get_loaders(
        root=DATA_DIR,
        download=False,
        batch_size=batch_size,
        num_workers=2,
        noise_type=noise_type,
        noise_rate=noise_rate
    )

    # Model, Loss, Optimizer
    model = create_model(depth=28, width_multiplier=2, num_classes=10, input_shape=input_shape, sgn=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # -------------------- CHANGES START HERE --------------------
    # Define warmup parameters
    base_lr = lr  # This is the lr you passed in, for example 0.1
    start_lr = base_lr * 0.1  # Starting at a lower LR for warmup (e.g. 0.01 if base_lr=0.1)

    # Create a cosine annealing scheduler for after warmup
    # Note: T_max is the number of epochs to run the cosine schedule
    # after warmup. So if total epochs = num_epochs, warmup is 5,
    # then T_max = num_epochs - warmup_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_epochs - warmup_epochs))

    # Define a helper function for warmup
    def warmup_lr_scheduler(optimizer, epoch, warmup_epochs, start_lr, base_lr):
        # Linearly scale LR from start_lr to base_lr over warmup_epochs
        lr = start_lr + (base_lr - start_lr) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # -------------------- CHANGES END HERE --------------------

    # Main Training Loop
    for epoch in range(num_epochs):
        # -------------------- CHANGES START HERE --------------------
        # Adjust the learning rate based on whether we're in warmup or not
        if epoch < warmup_epochs:
            # During warmup: manually set LR
            warmup_lr_scheduler(optimizer, epoch+1, warmup_epochs, start_lr, base_lr)
        else:
            # After warmup: use cosine annealing
            scheduler.step()
        # -------------------- CHANGES END HERE --------------------
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        result_dict['train_loss'].append(train_loss)
        result_dict['test_loss'].append(test_loss)
        result_dict['train_acc'].append(train_acc)
        result_dict['test_acc'].append(test_acc)


        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save checkpoint
        checkpoint_path = os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

    print("Training complete!")
    return result_dict