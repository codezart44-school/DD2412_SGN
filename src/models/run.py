import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from wide_resnet_28_2 import create_model

# Configuration
OUTPUT_DIR = './checkpoints'
DATA_DIR = './data'
BATCH_SIZE = 128
NUM_EPOCHS = 5
LEARNING_RATE = 0.1
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(SEED)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs, l2_penalty = model(images)  # Outputs and L2 penalty
        loss = criterion(outputs, labels) + l2_penalty  # Add L2 penalty to loss
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
            outputs, l2_penalty = model(images)  # Outputs and L2 penalty
            loss = criterion(outputs, labels) + l2_penalty  # Add L2 penalty to loss

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(loader), accuracy




def main():
    # Data loaders
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model, Loss, Optimizer
    model = create_model(depth=28, width_multiplier=2, num_classes=10, input_shape=train_dataset[0][0].shape, l2=5e-4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

    # Main Training Loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save checkpoint
        checkpoint_path = os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

    print("Training complete!")


if __name__=='__main__':
    main()